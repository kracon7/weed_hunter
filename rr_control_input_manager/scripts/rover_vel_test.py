#! /usr/bin/env python
import os
import sys
import argparse
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Int16MultiArray
from math import sin, cos, pi

def write_twist_msg(twist_msg, twist=[0.,0.,0.,0.,0.,0.]):
    twist_msg.linear.x = twist[0]
    twist_msg.linear.y = twist[1]
    twist_msg.linear.z = twist[2]
    twist_msg.angular.x = twist[3]
    twist_msg.angular.y = twist[4]
    twist_msg.angular.z = twist[5]
       
def main(args):
    '''
    First,
    Initialize the Gen3 cartesian space controler
    '''
    rospy.init_node('velocity_controller')
    rate=rospy.Rate(10)
    twist_pub = rospy.Publisher('cmd_vel/move_base', Twist, queue_size=1)
    twist_msg = Twist()

    time_start = rospy.Time.now()

    while not rospy.is_shutdown():

        time_now = rospy.Time.now()
        time_elap = time_now - time_start
        write_twist_msg(twist_msg, [0.1, 0., 0., 0., 0., 0.5])
        # if time_elap.to_sec < 5:
        #     write_twist_msg(twist_msg, [0.1, 0., 0., 0., 0., 0.5])
        # elif time_elap.to_sec < 10 and time_elap.to_sec < 5:
        #     write_twist_msg(twist_msg, [-0.2, 0., 0., 0., 0., 0.])
        # else:
        #     write_twist_msg(twist_msg, [0., 0., 0., 0., 0., 0.])
 
        twist_pub.publish(twist_msg)
        rate.sleep()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--velocity', default=0.01, type=float, help='cartesian velocity of end effector')
    parser.add_argument('--time_lim', default=4, type=float, help='robot action magnitude')
    args = parser.parse_args()
    
    main(args)