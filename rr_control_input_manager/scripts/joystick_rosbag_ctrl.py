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
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    if '/record_channel/cam_type' in rospy.get_param_names():
        cam_type = rospy.get_param('/record_channel/cam_type')
    else:
        cam_type = args.cam_type

    output_dir = args.output_dir

    print('\n\n\n', cam_type, '\n\n\n')
    if cam_type == 'single':
        rostopics = ['/camera/aligned_depth_to_color/camera_info',
                     '/camera/aligned_depth_to_color/image_raw',
                     '/camera/color/camera_info',
                     '/camera/color/image_raw',
                     '/rr_openrover_basic/odom_encoder',
                     '/tf',
                     '/tf_static']
    elif cam_type == "dual":
        rostopics = ['/d435/aligned_depth_to_color/camera_info',
                     '/d435/aligned_depth_to_color/image_raw',
                     '/d435/color/camera_info',
                     '/d435/color/image_raw',
                     '/d455/aligned_depth_to_color/camera_info',
                     '/d455/aligned_depth_to_color/image_raw',
                     '/d455/color/camera_info',
                     '/d455/color/image_raw',
                     '/d455/imu',
                     '/rr_openrover_basic/odom_encoder',
                     '/rr_openrover_basic/vel_calc_pub',
                     '/tf',
                     '/tf_static']   
    elif cam_type == 'shepherd':
        rostopics = ['/d435/aligned_depth_to_color/camera_info',
                     '/d435/aligned_depth_to_color/image_raw',
                     '/d435/color/camera_info',
                     '/d435/color/image_raw',
                     '/d435/infra1/camera_info',
                     '/d435/infra1/image_rect_raw',
                     '/d435/infra2/camera_info',
                     '/d435/infra2/image_rect_raw',
                     '/camera/aligned_depth_to_color/camera_info',
                     '/camera/aligned_depth_to_color/image_raw',
                     '/camera/color/camera_info',
                     '/camera/color/image_raw',
                     '/camera/imu',
                     '/rr_openrover_basic/odom_encoder',
                     '/tf',
                     '/tf_static']   
    elif cam_type == 'tri':
        rostopics = ['/d435/aligned_depth_to_color/camera_info',
                     '/d435/aligned_depth_to_color/image_raw',
                     '/d435/color/camera_info',
                     '/d435/color/image_raw',
                     '/d435/infra1/camera_info',
                     '/d435/infra1/image_rect_raw',
                     '/d435/infra2/camera_info',
                     '/d435/infra2/image_rect_raw',
                     '/front_d435/aligned_depth_to_color/camera_info',
                     '/front_d435/aligned_depth_to_color/image_raw',
                     '/front_d435/color/camera_info',
                     '/front_d435/color/image_raw',
                     '/front_d435/infra1/camera_info',
                     '/front_d435/infra1/image_rect_raw',
                     '/front_d435/infra2/camera_info',
                     '/front_d435/infra2/image_rect_raw',
                     '/camera/aligned_depth_to_color/camera_info',
                     '/camera/aligned_depth_to_color/image_raw',
                     '/camera/color/camera_info',
                     '/camera/color/image_raw',
                     '/camera/imu',
                     '/rr_openrover_basic/odom_encoder',
                     '/tf',
                     '/tf_static']

    # rostopics = ['/camera/aligned_depth_to_color/camera_info',
    #              '/camera/aligned_depth_to_color/image_raw',
    #              '/camera/color/camera_info',
    #              '/camera/color/image_raw']
    

    rospy.init_node("record_channel", anonymous=True)
    record_node = record_channel(output_dir, rostopics)
    record_node.start()


