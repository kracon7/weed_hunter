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
import subprocess, shlex


class record_channel(object):
    def __init__(self, output_dir, rostopics):
        self.is_recording = False
        self.loop_rate = rospy.Rate(100)
        rospy.Subscriber('/joystick/x_button', Bool, self.xbutton_cb, (output_dir, rostopics))
        rospy.Subscriber('/joystick/y_button', Bool, self.ybutton_cb)

        self.output_dir = output_dir
        self.rostopics = rostopics

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
            killcommand = "kill -9 " + str(self.p.pid)
            rospy.loginfo(killcommand)
            self.k = subprocess.Popen(killcommand, shell=True)
            rospy.loginfo("Recording has stopped")
            self.is_recording = False

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--output_dir', default='jiacheng/data', help='directory to store images')
    args = parser.parse_args()

    output_dir = args.output_dir
    rostopics = ['/camera/aligned_depth_to_color/camera_info',
                 '/camera/aligned_depth_to_color/image_raw',
                 '/camera/color/camera_info',
                 '/camera/color/image_raw',
                 '/rr_openrover_basic/odom_encoder']
    

    # rostopics = ['/camera/aligned_depth_to_color/camera_info',
    #              '/camera/aligned_depth_to_color/image_raw',
    #              '/camera/color/camera_info',
    #              '/camera/color/image_raw']
    

    rospy.init_node("record_channel_6", anonymous=True)
    record_node = record_channel(output_dir, rostopics)
    record_node.start()

