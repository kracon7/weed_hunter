#! /usr/bin/env python3
import os
import sys
import argparse
import rospy
import time
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from std_msgs.msg import Int32, Float32, Int16MultiArray
from ar_track_alvar_msgs.msg import AlvarMarkers
from math import sin, cos, pi

TARGETS = []

def ar_pose_cb(data, args):
    laser_pub = args[0]
    laser_msg = args[1]
    angle_servo_pub = args[2]
    angle_servo_msg = args[3]
    linear_servo_pub = args[4]
    linear_servo_msg = args[5]
    # if ar marker is detected
    if len(data.markers) > 0:
        # convert unit from m to mm
        x = 1000*data.markers[0].pose.pose.position.x
        y = 1000*data.markers[0].pose.pose.position.y
        z = 1000*data.markers[0].pose.pose.position.z
        target_position = np.array([x, y, z])
        angle, linear_servo_cmd = compute_laser_cmd(target_position)
        rospy.loginfo('Found target at {:.3f} degree and {:.3f} distance'.format(angle, linear_servo_cmd))
        if angle>=0 and angle<=25 and linear_servo_cmd>=0 and linear_servo_cmd<=180:
            angle_servo_msg.data = angle
            angle_servo_pub.publish(angle_servo_msg)
            linear_servo_msg.data = linear_servo_cmd
            linear_servo_pub.publish(linear_servo_msg)
            laser_msg.data = 1
            laser_pub.publish(laser_msg)
        else:
            laser_msg.data = 0
            laser_pub.publish(laser_msg)

def compute_laser_cmd(target_position):
    origin = np.array([37.87, -62.54, -141.75])
    linear_axis = np.array([0.9988,  0.0192,  -0.0451])
    x_axis = np.array([ 0.9995, -0.0190, -0.0235])
    y_axis = np.array([ 0.0171,  0.9969, -0.0768])
    z_axis = np.array([ 0.0249,  0.0763,  0.9968])

    print('target_position: ', target_position)

    linear_servo_length = (target_position[0] - origin[0]) 
    linear_servo_cmd = linear_servo_length / 25.4 / 6 * 180

    # compute the laser angle corresponding to the projected target vector 
    vec = target_position - np.array([target_position[0], origin[1], origin[2]])
    print('vec: ', vec)
    # projected_vec = vec - (np.dot(vec, x_axis) * x_axis / np.linalg.norm(vec)) 
    # # print(projected_vec)
    # rad = np.arccos(np.dot(projected_vec, z_axis) / np.linalg.norm(projected_vec))
    rad = np.arccos(np.dot(vec, np.array([0,0,1])) / np.linalg.norm(vec))
    angle = rad/pi*180

    # compute the distance between target and laser x-y plane
    dist = np.dot(vec, z_axis)
    return angle, linear_servo_cmd
       
def main(args):
    '''
    First,
    Initialize the Gen3 cartesian space controler
    '''
    rospy.init_node('laser_controller')
    rate=rospy.Rate(10)

    laser_pub = rospy.Publisher('/laser_cmd', Int32, queue_size=1)
    laser_msg = Int32()

    angle_servo_pub = rospy.Publisher('/servo_angle', Float32, queue_size=1)
    angle_servo_msg = Float32()
    
    linear_servo_pub = rospy.Publisher('/servo_length', Float32, queue_size=1)
    linear_servo_msg = Float32()

    ar_pose_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, ar_pose_cb, 
                                        [laser_pub, laser_msg, angle_servo_pub, angle_servo_msg, 
                                         linear_servo_pub, linear_servo_msg])

    time_start = rospy.Time.now()


    while not rospy.is_shutdown():
        # target_position, mask = process_frame(color_image, depth_image)
        # print('Found target position in camera frame: {}'.format(target_position))
        # angle = compute_laser_cmd(target_position)
        # print('Laser angle command: {}'.format(angle))

        # if angle > 0. and angle < 20.:
        #     servo_msg.data = angle
        #     servo_pub.publish(servo_msg)
        #     laser_msg.data = 1
        #     laser_pub.publish(laser_msg)
        rate.sleep()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--timeout', default=40, type=float, help='total time of image streaming')
    parser.add_argument('--output_dir', default='data', help='directory to store images')
    parser.add_argument('--theta', type=str, default='0', help='angle of servo')
    args = parser.parse_args()
    
    main(args)