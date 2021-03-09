#! /usr/bin/env python3
import os
import sys
import argparse
import rospy
import numpy as np
import pyrealsense2
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Int16MultiArray
from math import sin, cos, pi

       
def main(args):
    '''
    First,
    Initialize the Gen3 cartesian space controler
    '''
    rospy.init_node('velocity_controller')
    rate=rospy.Rate(10)
    test_pub = rospy.Publisher('test', Int32, queue_size=1)
    test_msg = Int32()

    time_start = rospy.Time.now()

    while not rospy.is_shutdown():

        
        test_msg.data = 1
        test_pub.publish(test_msg)
        print('here ')
        rate.sleep()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--velocity', default=0.01, type=float, help='cartesian velocity of end effector')
    parser.add_argument('--time_lim', default=4, type=float, help='robot action magnitude')
    args = parser.parse_args()
    
    main(args)