#! /home/jc/Envs/py36/bin/python3.6

import os
import sys
import argparse
import rospy
import struct
import time
import numpy as np
import datetime as dt
from std_msgs.msg import Int32, Float32, Bool, Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import message_filters

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard")
    print(data.header)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/front_d435/color/image_raw", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()