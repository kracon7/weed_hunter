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

class FrameListener():
    """docstring for frame_listener"""
    def __init__(self):
        self.prev_pose = PoseStamped()
        self.prev_pose.header.stamp = None
        self.img_sub = rospy.Subscriber("/d435/color/image_raw", Image, self.callback)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pos = [0, 0, 0]

    def callback(self, data):

        # rospy.loginfo(rospy.get_caller_id() + "I heard")
        
        frame_id = data.header.frame_id
        # print("data timestamp is: %d, %d"%(data.header.stamp.secs, data.header.stamp.nsecs))

        if self.prev_pose.header.stamp is not None:
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

    
def listener():

    rospy.init_node('listener', anonymous=True)

    frame_listener = FrameListener()

    rospy.spin()

if __name__ == '__main__':
    listener()