#!/home/jc/Envs/py36/bin/python3.6

import os
import sys
import argparse
import rospy
import struct
import time
import pickle
import numpy as np
import datetime as dt
import open3d as o3d
from std_msgs.msg import Int32, Float32, Bool, Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from rr_control_input_manager.msg import CornPlane
import message_filters

name_to_dtypes = {
    "rgb8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "rgb16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgr8":    (np.uint8,  3),
    "bgra8":   (np.uint8,  4),
    "bgr16":   (np.uint16, 3),
    "bgra16":  (np.uint16, 4),
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),
    
    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8":  (np.uint8,  1),
    "bayer_bggr8":  (np.uint8,  1),
    "bayer_gbrg8":  (np.uint8,  1),
    "bayer_grbg8":  (np.uint8,  1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1":    (np.uint8,   1),
    "8UC2":    (np.uint8,   2),
    "8UC3":    (np.uint8,   3),
    "8UC4":    (np.uint8,   4),
    "8SC1":    (np.int8,    1),
    "8SC2":    (np.int8,    2),
    "8SC3":    (np.int8,    3),
    "8SC4":    (np.int8,    4),
    "16UC1":   (np.uint16,   1),
    "16UC2":   (np.uint16,   2),
    "16UC3":   (np.uint16,   3),
    "16UC4":   (np.uint16,   4),
    "16SC1":   (np.int16,  1),
    "16SC2":   (np.int16,  2),
    "16SC3":   (np.int16,  3),
    "16SC4":   (np.int16,  4),
    "32SC1":   (np.int32,   1),
    "32SC2":   (np.int32,   2),
    "32SC3":   (np.int32,   3),
    "32SC4":   (np.int32,   4),
    "32FC1":   (np.float32, 1),
    "32FC2":   (np.float32, 2),
    "32FC3":   (np.float32, 3),
    "32FC4":   (np.float32, 4),
    "64FC1":   (np.float64, 1),
    "64FC2":   (np.float64, 2),
    "64FC3":   (np.float64, 3),
    "64FC4":   (np.float64, 4)
}



class FitGround(object):
    def __init__(self, args):
        self.args = args
        self.loop_rate = rospy.Rate(100)

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        color_sub = message_filters.Subscriber('/front_d435/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/front_d435/aligned_depth_to_color/image_raw', Image)
        side_sub = message_filters.Subscriber('/d435/color/image_raw', Image)
        # ts = message_filters.TimeSynchronizer([color_sub, depth_sub, info_sub], 1)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, side_sub], 1, 1)
        ts.registerCallback(self.callback)

        self.im_w = 848
        self.im_h = 480
        self.K = np.array([[615.311279296875,   0.0,             430.1778869628906],
                           [  0.0,            615.4699096679688, 240.68307495117188],
                           [  0.0,              0.0,               1.0]])
        # compute rays in advance
        x, y = np.arange(self.im_w), np.arange(self.im_h)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=2).reshape(-1,2)
        self.rays = np.dot(np.insert(points, 2, 1, axis=1), np.linalg.inv(self.K).T).reshape(self.im_h, self.im_w, 3)
        rospy.loginfo('Ground fit initialization finished...')

    def callback(self, color, depth, side):
        # Solve all of perception here...
        t_sec = color.header.stamp.secs
        rospy.loginfo('Received frame at time: %d'%(t_sec))

        np_color = self.image_to_numpy(color)
        np_depth = self.image_to_numpy(depth).astype('float32') * 1e-3
        np_side_color = self.image_to_numpy(side)
        assert (np_color.shape[0] == self.im_h) and (np_color.shape[1] == self.im_w), \
                'Image size incorrect, expected %d, %d but got %d, %d instead'%(self.im_h, self.im_w, np_color.shape[0], np.color.shape[1])

        points = self.rays * np_depth.reshape(self.im_h, self.im_w, 1)
        xyzrgb = np.concatenate([points, np_color], axis=2)

        fname = '%d_%d.npy'%(color.header.stamp.secs, color.header.stamp.nsecs)
        np.save(os.path.join(self.args.output_dir, 'front_rgbd', fname), xyzrgb)
        np.save(os.path.join(self.args.output_dir, 'side', fname), np_side_color)


    def image_to_numpy(self, msg):
        if not msg.encoding in name_to_dtypes:
            raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
        
        dtype_class, channels = name_to_dtypes[msg.encoding]
        dtype = np.dtype(dtype_class)
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
        shape = (msg.height, msg.width, channels)

        data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
        data.strides = (
            msg.step,
            dtype.itemsize * channels,
            dtype.itemsize
        )

        if channels == 1:
            data = data[...,0]
        return data
    
    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--output_dir', default='/home/jc/tmp', help='directory to store images')
    args, unknown = parser.parse_known_args()

    rospy.init_node("extract_bag_sync", anonymous=True)
    fit_node = FitGround(args)
    fit_node.start()


