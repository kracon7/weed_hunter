#! /home/jc/Envs/py36/bin/python3.6

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

    def callback(self, data):

        rospy.loginfo(rospy.get_caller_id() + "I heard")
        
        frame_id = data.header.frame_id

        if self.prev_pose.header.stamp is not None:
            transform = self.tfBuffer.lookup_transform(frame_id, self.prev_pose.header.stamp, 
                            frame_id, data.header.stamp)
            
            # store prev_pose for next frame
            self.prev_pose.stamp = data.header.stamp
            self.prev_pose.position.x = transform.transform.translation.x
            self.prev_pose.position.y = transform.transform.translation.y
            self.prev_pose.position.z = transform.transform.translation.z
            self.prev_pose.orientation.x = transform.transform.rotation.x
            self.prev_pose.orientation.y = transform.transform.rotation.y
            self.prev_pose.orientation.z = transform.transform.rotation.z
            self.prev_pose.orientation.w = transform.transform.rotation.w


    
def listener():

    rospy.init_node('listener', anonymous=True)

    frame_listener = FrameListener()

    rospy.spin()

if __name__ == '__main__':
    listener()