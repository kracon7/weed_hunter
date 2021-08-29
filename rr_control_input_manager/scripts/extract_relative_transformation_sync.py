#! /usr/bin/env python2.7

import os
import sys
import argparse
import rospy
import struct
import time
import numpy as np
import datetime as dt
import tf2_ros
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, Float32, Bool, Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import Odometry
import message_filters
from utils.cv_bridge import image_to_numpy

class FrameListener():
    """docstring for frame_listener"""
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.front_rgbd_dir = os.path.join(args.output_dir, 'front_rgbd')
        self.side_color_dir = os.path.join(args.output_dir, 'side_color')
        self.transform_dir = os.path.join(args.output_dir, 'transform')
        for d in [self.front_rgbd_dir, self.side_color_dir, self.transform_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        self.prev_pose = PoseStamped()
        self.prev_pose.header.stamp = None
        color_sub = message_filters.Subscriber('/front_d435/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/front_d435/aligned_depth_to_color/image_raw', Image)
        side_sub = message_filters.Subscriber("/d435/color/image_raw", Image)
        
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, side_sub], 1, 1)
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
        

    def callback(self, front_color, front_depth, data):

        # rospy.loginfo(rospy.get_caller_id() + "I heard")
        
        frame_id = data.header.frame_id
        print("data timestamp is: %d, %d"%(data.header.stamp.secs, data.header.stamp.nsecs))

        if self.prev_pose.header.stamp is not None:
            try:
                transform = self.tfBuffer.lookup_transform(frame_id, 'map',
                                         data.header.stamp)

                # compute the relative transformation
                rel_x = - transform.transform.translation.x + self.prev_pose.pose.position.x
                rel_y = - transform.transform.translation.y + self.prev_pose.pose.position.y
                rel_z = - transform.transform.translation.z + self.prev_pose.pose.position.z
                quat_now = np.array([transform.transform.rotation.w,
                                     transform.transform.rotation.x,
                                     transform.transform.rotation.y,
                                     transform.transform.rotation.z])
                quat_prev = np.array([self.prev_pose.pose.orientation.w,
                                      self.prev_pose.pose.orientation.x,
                                      self.prev_pose.pose.orientation.y,
                                      self.prev_pose.pose.orientation.z])

                rel_quat = self.quaternion_division(quat_prev, quat_now)

                # store prev_pose for next frame
                self.prev_pose.header.stamp = data.header.stamp
                self.prev_pose.pose.position.x = transform.transform.translation.x
                self.prev_pose.pose.position.y = transform.transform.translation.y
                self.prev_pose.pose.position.z = transform.transform.translation.z
                self.prev_pose.pose.orientation.x = transform.transform.rotation.x
                self.prev_pose.pose.orientation.y = transform.transform.rotation.y
                self.prev_pose.pose.orientation.z = transform.transform.rotation.z
                self.prev_pose.pose.orientation.w = transform.transform.rotation.w

                # print("relative transformation: %f %f %f %f %f %f %f"%(
                #         rel_x, rel_y, rel_z, rel_quat[0], rel_quat[1], rel_quat[2], rel_quat[3]))
                self.pos[0] += rel_x
                self.pos[1] += rel_y
                self.pos[2] += rel_z
                print("x y z: %f %f %f "%(self.pos[0], self.pos[1], self.pos[2]))

                self.count += 1

                # convert images to numpy array
                np_front_color = image_to_numpy(front_color)
                np_front_depth = image_to_numpy(front_depth).astype('float32') * 1e-3
                assert (np_front_color.shape[0] == self.im_h) and (np_front_color.shape[1] == self.im_w), \
                        'Image size incorrect, expected %d, %d but got %d, %d instead'%(self.im_h, self.im_w, np_color.shape[0], np.color.shape[1])

                points = self.rays * np_front_depth.reshape(self.im_h, self.im_w, 1)
                xyzrgb = np.concatenate([points, np_front_color], axis=2)
                np.save(os.path.join(self.front_rgbd_dir, 'frame_%07d.npy'%(self.count)), xyzrgb)

                np_side_color = image_to_numpy(data)
                np.save(os.path.join(self.side_color_dir, 'frame_%07d.npy'%(self.count)), np_side_color)

                # save relative transformation
                T = np.concatenate([np.array([rel_x, rel_y, rel_z]), rel_quat])
                np.save(os.path.join(self.transform_dir, 'frame_%07d.npy'%(self.count)), T)

            except Exception as e:
                print(e)
            
        else:
            self.prev_pose.header.stamp = data.header.stamp

    def quaternion_division(self, q, r):
        qw, qx, qy, qz = q
        rw, rx, ry, rz = r

        tw = rw*qw + rx*qx + ry*qy + rz*qz
        tx = rw*qx - rx*qw - ry*qz + rz*qy
        ty = rw*qy + rx*qz - ry*qw - rz*qx
        tz = rw*qz - rx*qy + ry*qx - rz*qw
        return [tw, tx, ty, tz]

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--output_dir', default='/home/jc/tmp/pred_distance', help='directory to store images')
    args, unknown = parser.parse_known_args()

    rospy.init_node('listener', anonymous=True)

    frame_listener = FrameListener(args)

    rospy.spin()
