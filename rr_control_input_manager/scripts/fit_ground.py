#! /home/rsn/Env/py35/bin/python3.5

import os
import sys
import argparse
import rospy
import time
import multiprocessing as mp
import threading
from pathlib import Path
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from std_msgs.msg import Int32, Float32, Bool
from sensor_msgs.msg import Image, CameraInfo
import subprocess, shlex
import message_filters


class FitGround(object):
    def __init__(self):
        self.loop_rate = rospy.Rate(100)

        color_sub = message_filters.Subscriber('/front_d435/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/front_d435/aligned_depth_to_color/image_raw', Image)
        info_sub = message_filters.Subscriber('/front_d435/aligned_depth_to_color/camera_info', CameraInfo)

        ts = message_filters.TimeSynchronizer([color_sub, depth_sub, info_sub], 1)
        ts.registerCallback(self.callback)

    def callback(self, color, depth, camera_info):
        # Solve all of perception here...
        time = camera_info.header.stamp.secs
        rospy.loginfo('Received frame at time: %d'%(time))


    def xbutton_cb(self, data, args):
        output_dir = args[0]
        rostopics = args[1]
        
        if self.is_recording == False:

            # rosbag recording command
            t = dt.datetime.now()
            HOME = str(Path.home())
            bag_path = os.path.join(HOME, output_dir, '%d-%d_%d-%d.bag'%(t.month, t.day, t.hour, t.minute))
            cmd = "rosbag record -O %s"%bag_path

            for topic in self.rostopics:
                cmd += ' %s'%topic
            cmd += " __name:=bag_record"
            print(cmd)

            self.p = subprocess.Popen(shlex.split(cmd))
            self.is_recording = True
            rospy.loginfo('Process ID: %d, Recording data....'%(self.p.pid))
    
    def ybutton_cb(self, data):
        if self.is_recording:
            rospy.loginfo('Killing the recording process...')
            # killcommand = "kill -9 " + str(self.p.pid)
            # rospy.loginfo(killcommand)
            # self.k = subprocess.Popen(killcommand, shell=True)
            subprocess.Popen(shlex.split("rosnode kill /bag_record"))
            rospy.loginfo("Recording has stopped")
            self.is_recording = False

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


