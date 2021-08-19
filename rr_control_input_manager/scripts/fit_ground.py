#!/home/jc/Envs/py36/bin/python3.6

import os
import sys
import argparse
import rospy
import struct
import time
import numpy as np
import datetime as dt
import open3d as o3d
from std_msgs.msg import Int32, Float32, Bool, Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from rr_control_input_manager.msg import CornPlane
import message_filters

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils.model import CorridorNet

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
    def __init__(self):
        self.loop_rate = rospy.Rate(100)

        color_sub = message_filters.Subscriber('/front_d435/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/front_d435/aligned_depth_to_color/image_raw', Image)
        # self.filtered_color_pub = rospy.Publisher('/filtered_color', Image)
        # self.filtered_color_msg = Image()
        self.pc2_pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)

        # publish corn plane message
        self.plane_pub = rospy.Publisher("corn_plane", CornPlane, queue_size=1)
        self.plane_msg = CornPlane()

        # ts = message_filters.TimeSynchronizer([color_sub, depth_sub, info_sub], 1)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 1, 1)
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

        rospy.loginfo('Loading network module...')
        self.corridor_net = CorridorNet().to('cuda')

    def callback(self, color, depth):
        # Solve all of perception here...
        t_sec = color.header.stamp.secs
        rospy.loginfo('Received frame at time: %d'%(t_sec))

        np_color = self.image_to_numpy(color)
        np_depth = self.image_to_numpy(depth).astype('float32') * 1e-3
        assert (np_color.shape[0] == self.im_h) and (np_color.shape[1] == self.im_w), \
                'Image size incorrect, expected %d, %d but got %d, %d instead'%(self.im_h, self.im_w, np_color.shape[0], np.color.shape[1])

        # color thresholding to find ground points
        mask = (np_color[:,:,1] > 0.5*np_color[:,:,0]) & (np_color[:,:,1] <= 0.95*np_color[:,:,0]) & \
               (np_color[:,:,1] > 0.6*np_color[:,:,2]) & (np_color[:,:,1] <= 0.9*np_color[:,:,2]) & \
               (np_color[:,:,2] <= 100)

        idx = np.where(mask)
        ground_points = self.rays[idx[0], idx[1]] * np_depth[idx[0], idx[1]].reshape(-1,1)
        
        # fit ground points to find plane
        now = time.time()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))
        plane_model, _ = pcd.segment_plane(distance_threshold=0.03,
                                             ransac_n=3,
                                             num_iterations=100)
        print("Took %.5fsec to find plane %.3f %.3f %.3f %.3f"%(
                time.time()-now, plane_model[0], plane_model[1], plane_model[2], plane_model[3]))
        
        # run network on rgb image to predict vanishing point and corn lines
        now = time.time()
        torch_color = torch.from_numpy(np_color.transpose((2,0,1))).float().unsqueeze(0).cuda()
        pred = self.corridor_net.forward(torch_color).view(-1)
        print("Took %.5fsec to predict %.3f %.3f %.3f %.3f"%(
                time.time()-now, pred[0], pred[1], pred[2], pred[3]))

        # write corn plane message and publish
        self.plane_msg.header.stamp = color.header.stamp
        self.plane_msg.pt.x = 0.1
        self.plane_msg.pt.y = 0.2
        self.plane_msg.pt.z = 0.1
        self.plane_msg.vy.x = 0.
        self.plane_msg.vy.y = 1.
        self.plane_msg.vy.z = 0.
        self.plane_msg.vz.x = 0.
        self.plane_msg.vz.y = 0.
        self.plane_msg.vz.z = 1.
        self.plane_pub.publish(self.plane_msg)
        

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

    def points_to_pointcloud(self, pts, colors):
        points = []
        
        for pt, color in zip(pts, colors):
            x, y, z = pt
            r, g, b = color
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            points.append([x, y, z, rgb])

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1),
                  ]

        header = Header()
        header.frame_id = "front_d435_link"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        return pc2
    
    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--output_dir', default='jiacheng/data', help='directory to store images')
    parser.add_argument('--cam_type', default='shepherd', type=str, help='type of camera setup, dual or single')
    args, unknown = parser.parse_known_args()

    rospy.init_node("fit_ground", anonymous=True)
    fit_node = FitGround()
    fit_node.start()


