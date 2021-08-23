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

        rospy.loginfo("Loading network parameters...")
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils/corridor_net_chckpt.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        self.corridor_net.load_state_dict(checkpoint['state_dict'])

        rospy.loginfo("Loading network prediction stat...")
        stat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils/corridor_stat.pkl')
        self.stat = pickle.load(open(stat_path,'rb'))

        rospy.loginfo("Initialization Done!")

    def callback(self, color, depth):
        # Solve all of perception here...
        t_sec = color.header.stamp.secs
        rospy.loginfo('Received frame at time: %d'%(t_sec))

        np_color = self.image_to_numpy(color)
        np_depth = self.image_to_numpy(depth).astype('float32') * 1e-3
        assert (np_color.shape[0] == self.im_h) and (np_color.shape[1] == self.im_w), \
                'Image size incorrect, expected %d, %d but got %d, %d instead'%(self.im_h, self.im_w, np_color.shape[0], np.color.shape[1])

        points = self.rays * np_depth.reshape(self.im_h, self.im_w, 1)

        # color thresholding to find ground points
        mask = (np_color[:,:,1] > 0.5*np_color[:,:,0]) & (np_color[:,:,1] <= 0.95*np_color[:,:,0]) & \
               (np_color[:,:,1] > 0.6*np_color[:,:,2]) & (np_color[:,:,1] <= 0.9*np_color[:,:,2]) & \
               (np_color[:,:,2] <= 100)
        idx = np.where(mask)
        # ground_points = self.rays[idx[0], idx[1]] * np_depth[idx[0], idx[1]].reshape(-1,1)
        ground_points = points[idx[0], idx[1]]
        
        # fit ground points to find plane
        now = time.time()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))
        plane_model, _ = pcd.segment_plane(distance_threshold=0.03,
                                             ransac_n=3,
                                             num_iterations=100)
        # if surface normal direction points downward
        if plane_model[1] < 0:
            plane_model = - plane_model
        print("Took %.5fsec to find plane %.3f %.3f %.3f %.3f"%(
                time.time()-now, plane_model[0], plane_model[1], plane_model[2], plane_model[3]))
        
        # run network on rgb image to predict vanishing point and corn lines
        now = time.time()
        torch_color = torch.from_numpy(np_color.transpose((2,0,1))).float().unsqueeze(0).cuda()
        pred = self.corridor_net.forward(torch_color).view(-1).detach().cpu().numpy()
        pred = pred * self.stat['std'] + self.stat['mean']
        print("Took %.5fsec to predict %.3f %.3f %.3f %.3f"%(
                time.time()-now, pred[0], pred[1], pred[2], pred[3]))

        # use height thresholding to find corridor points
        mask = (points[:,:,0] > -0.5) & (points[:,:,0] < 0.5) & \
               (points[:,:,1] > -0.05) & (points[:,:,1] < -plane_model[3]-.05) & \
               (points[:,:,2] > 0.1) & (points[:,:,2] < 3)
        idx = np.where(mask)
        corridor_points = points[idx[0], idx[1]]

        # paint ground green and paint corridor red
        ground_color = np.zeros_like(ground_points).astype('uint8')
        ground_color[:,0] = 255
        corridor_color = np.zeros_like(corridor_points).astype('uint8')
        corridor_color[:,1] = 255

        pcd = self.points_to_pointcloud(np.concatenate([ground_points, corridor_points], axis=0),
                            np.concatenate([ground_color, corridor_color], axis=0))
        self.pc2_pub.publish(pcd)


    def publish_plane(self, stamp, pt, vy, vz):
        '''
        Publish the corn plane in front camera frame
        '''
        # TODO check unlikely cases of the plane model

        # write corn plane message and publish
        self.plane_msg.header.stamp = stamp
        self.plane_msg.pt.x = pt[0]
        self.plane_msg.pt.y = pt[1]
        self.plane_msg.pt.z = pt[2]
        self.plane_msg.vy.x = vy[0]
        self.plane_msg.vy.y = vy[1]
        self.plane_msg.vy.z = vy[2]
        self.plane_msg.vz.x = vz[0]
        self.plane_msg.vz.y = vz[1]
        self.plane_msg.vz.z = vz[2]
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


