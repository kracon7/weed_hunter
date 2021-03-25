#! /usr/bin/env python3
import os
import sys
import argparse
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from opencv_apps.msg import LineArrayStamped
from std_msgs.msg import Int32, Int16MultiArray
from math import sin, cos, pi

def write_twist_msg(twist_msg, twist=[0.,0.,0.,0.,0.,0.]):
    twist_msg.linear.x = twist[0]
    twist_msg.linear.y = twist[1]
    twist_msg.linear.z = twist[2]
    twist_msg.angular.x = twist[3]
    twist_msg.angular.y = twist[4]
    twist_msg.angular.z = twist[5]


def extract_lines(hough_lines):
    N = len(hough_lines)
    extracted = np.zeros((N, 6))

    for i in range(N):
        pt1x = hough_lines[i].pt1.x
        pt1y = hough_lines[i].pt1.y
        pt2x = hough_lines[i].pt2.x
        pt2y = hough_lines[i].pt2.y
        extracted[i, :4] = np.array([pt1x, pt1y, pt2x, pt2y])

        theta = np.rad2deg(np.arctan2(pt2y - pt1y, pt2x - pt1x))
        if theta <=0:
            theta += 90
        else:
            theta -= 90
        length = np.linalg.norm(np.array([pt2y - pt1y, pt2x - pt1x]))

        extracted[i, 4:6] = np.array([theta, length])

    mask = extracted[:, 5] > 200
    extracted = extracted[mask]

    if extracted.shape[0] > 1:
        # sort according to line length and pick the largest two
        order = np.argsort(extracted[:, 5])
        extracted = extracted[order[:1]]

    return extracted
       

def hough_line_cb(data, args):
    twist_pub = args[0]
    twist_msg = args[1]

    extracted = extract_lines(data.lines)
    # print(extracted)

    if extracted.shape[0] > 0:
        line_angle = np.average(extracted[:,4])
        print(line_angle)


def main(args):
    '''
    First,
    Initialize the Gen3 cartesian space controler
    '''
    rospy.init_node('velocity_controller')
    rate=rospy.Rate(10)

    twist_pub = rospy.Publisher('cmd_vel/move_base', Twist, queue_size=1)
    twist_msg = Twist()


    hough_line_sub = rospy.Subscriber('/hough_lines/lines', LineArrayStamped, hough_line_cb, 
                                        [twist_pub, twist_msg])

    time_start = rospy.Time.now()

    while not rospy.is_shutdown():

        # time_now = rospy.Time.now()
        # time_elap = time_now - time_start
        # write_twist_msg(twist_msg, [0.1, 0., 0., 0., 0., 0.5])
        # if time_elap.to_sec < 5:
        #     write_twist_msg(twist_msg, [0.1, 0., 0., 0., 0., 0.5])
        # elif time_elap.to_sec < 10 and time_elap.to_sec < 5:
        #     write_twist_msg(twist_msg, [-0.2, 0., 0., 0., 0., 0.])
        # else:
        #     write_twist_msg(twist_msg, [0., 0., 0., 0., 0., 0.])
 
        # twist_pub.publish(twist_msg)
        rate.sleep()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--velocity', default=0.01, type=float, help='cartesian velocity of end effector')
    parser.add_argument('--time_lim', default=4, type=float, help='robot action magnitude')
    args = parser.parse_args()
    
    main(args)