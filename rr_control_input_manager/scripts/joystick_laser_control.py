#! /home/rsn/Env/py35/bin/python3.5
import os
import sys
import argparse
import rospy
import time
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from std_msgs.msg import Int32, Float32, Bool

TARGETS = []

def joystick_cb(data):
    estop = rospy.get_param('/laser_estop')

    if estop:
        estop = False
        rospy.loginfo('Laser Estop turned off')
        rospy.set_param('laser_estop', estop)
    else:
        estop = True
        rospy.loginfo('Laser Estop is ON, press A to switch')
        rospy.set_param('laser_estop', estop)
        
       
def main(args):
    '''
    First,
    Initialize the Gen3 cartesian space controler
    '''
    rospy.init_node('joystick_laser_control')
    rate=rospy.Rate(10)

    rospy.set_param('laser_estop', False)

    joystick_sub = rospy.Subscriber('/joystick/a_button', Bool, joystick_cb)

    while not rospy.is_shutdown():
        rate.sleep()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    args = parser.parse_args()
    
    main(args)