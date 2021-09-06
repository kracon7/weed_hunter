#!/home/jc/Envs/py36/bin/python3.6

import os
import sys
import argparse
import rospy
import struct
import time
import cv2
import numpy as np
import datetime as dt
import tf2_ros
import pickle
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, Float32, Bool, Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import Odometry
import message_filters
from utils.cv_bridge import image_to_numpy


class Frame():
    """sync-ed frame for side and front view"""
    def __init__(self, front_xyzrgb, side_color, stamp, pose):
        self.front_xyzrgb = front_xyzrgb
        self.side_color = side_color
        self.stamp = stamp
        self.pose = pose

class FrameListener():
    """docstring for frame_listener"""
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.front_rgbd_dir = os.path.join(args.output_dir, 'front_rgbd')
        self.side_color_dir = os.path.join(args.output_dir, 'side_color')
        self.frame_dir = os.path.join(args.output_dir, 'frame')
        for d in [self.front_rgbd_dir, self.side_color_dir, self.frame_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        self.prev_pose = PoseStamped()
        self.prev_pose.header.stamp = None
        color_sub = message_filters.Subscriber('/front_d435/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/front_d435/aligned_depth_to_color/image_raw', Image)
        side_sub = message_filters.Subscriber("/d435/color/image_raw", Image)
        
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, side_sub], 100, 1)
        ts.registerCallback(self.callback)

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(20))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pos = [0, 0, 0]
        self.count = 0

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
        rospy.loginfo('Initialization done!')

    def callback(self, front_color, front_depth, side_color):
        
        frame_id = side_color.header.frame_id
        t_sec = side_color.header.stamp.secs
        t_nsec = side_color.header.stamp.nsecs
        rospy.loginfo('Received frame with stamp: %d.%d'%(t_sec, t_nsec))

        try:
            transform = self.tfBuffer.lookup_transform(frame_id, 'map',
                                     side_color.header.stamp)

            pose = np.array([transform.transform.translation.x,
                             transform.transform.translation.y,
                             transform.transform.translation.z,
                             transform.transform.rotation.w,
                             transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z])
            self.count += 1

            # convert images to numpy array
            np_front_color = image_to_numpy(front_color)
            np_front_depth = image_to_numpy(front_depth).astype('float32') * 1e-3
            np_side_color = image_to_numpy(side_color)
            
            points = self.rays * np_front_depth.reshape(self.im_h, self.im_w, 1)
            front_xyzrgb = np.concatenate([points, np_front_color], axis=2)

            frame = Frame(front_xyzrgb, np_side_color, side_color.header.stamp, pose)

            cv2.imwrite(os.path.join(self.front_rgbd_dir, 'frame_%07d.png'%(self.count)), np_front_color)
            cv2.imwrite(os.path.join(self.side_color_dir, 'frame_%07d.png'%(self.count)), np_side_color)
            pickle.dump(frame, open(os.path.join(self.frame_dir, 'frame_%07d.pkl'%(self.count)),'wb'))

        except Exception as e:
            print(e)
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--output_dir', default='/home/jc/tmp/pred_distance', help='directory to store images')
    args, unknown = parser.parse_known_args()

    rospy.init_node('listener', anonymous=True)

    frame_listener = FrameListener(args)

    rospy.spin()
