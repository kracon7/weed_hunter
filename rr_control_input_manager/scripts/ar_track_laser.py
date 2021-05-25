#! /home/rsn/Env/py35/bin/python3.5
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
from sympy import *

TARGETS = []

def ar_pose_cb(data, args):
    servo1_pub = args[0]
    servo1_msg = args[1]
    servo2_pub = args[2]
    servo2_msg = args[3]
    # if ar marker is detected
    if len(data.markers) > 0:
        # convert unit from m to mm
        x = 1000*data.markers[0].pose.pose.position.x
        y = 1000*data.markers[0].pose.pose.position.y
        z = 1000*data.markers[0].pose.pose.position.z
        target_position = np.array([x, y, z])
        angle_1, angle_2 = compute_laser_cmd(target_position)
        
        # if target is within the range of the actuator
        if angle_1 is not None:
            rospy.loginfo('Found target at {:.3f} degree and {:.3f} degree'.format(angle_1, angle_2))
            
            # if target if close to the laser plane, turn the laser off
            if np.abs(angle_1 - 30) < 5:
                rospy.loginfo('Target close to laser plane, turnning laser off')
                # servo1_msg.data = 30
                # servo1_pub.publish(servo2_msg)
                servo2_msg.data = angle_2
                servo2_pub.publish(servo2_msg)
                rospy.set_param('laser_cmd', 0)
            else:
                rospy.set_param('laser_cmd', 1)
        # else:
        #     print('No valid sol found')
        #     rospy.set_param('laser_cmd', 1)

    else:
        rospy.set_param('laser_cmd', 1)
                

def compute_laser_cmd(target):
    # rotation axis e1
    e1 = np.array([0.0269, -0.7183, -0.6952])
    # reverse of x_axis vx
    vx = np.array([-0.8163, -0.4171, 0.3995])
    vz = np.array([0.420562588779785, 0.0447626641885372, 0.906158602460734])
    vy = np.cross(vz, vx)

    theta = symbols('theta')
    L = 59.7

    origin = np.array([32.4282, -81.5175, -147.657])
    V = target - origin

    expr_1 = vx * cos(theta) + np.cross(e1, vx) * sin(theta) + e1 @ vx * e1 * (1-cos(theta))
    eq1 = Eq(expr_1 @ V, L)
    result = solve(eq1, theta)

    if len(result) > 0:
        for th1 in result:
            angle_1 = np.rad2deg(float(th1))
            
            # if valid angle for rotation 1 is found
            if angle_1 > 0 and angle_1 < 60:
                theta_1 = float(th1)

                # keep solving for theta_2
                rotated_x = np.zeros(3)
                for i in range(3):
                    rotated_x[i] = expr_1[i].subs(theta, theta_1)

                rotated_z = vz*np.cos(theta_1) + np.cross(e1, vz)*np.sin(theta_1) \
                            + e1@vz * e1 * (1-np.cos(theta_1))
                rotated_y = vy*np.cos(theta_1) + np.cross(e1, vy)*np.sin(theta_1) \
                            + e1@vy * e1 * (1-np.cos(theta_1))

                origin_2 = origin + L * rotated_x
                vector = target - origin_2

                # project vector to the plane of z_axis and y_axis
                vector = vector - vector @ rotated_x * rotated_x
                vector = vector / np.linalg.norm(vector)

                angle_2 = np.rad2deg(np.arctan2(rotated_z @ vector, rotated_y @ vector)) - 90
                
                if angle_2 >= 0 and angle_2 <= 20:  
                    return angle_1, angle_2

    return None, None

       
def main(args):
    '''
    First,
    Initialize the Gen3 cartesian space controler
    '''
    rospy.init_node('laser_controller')
    rate=rospy.Rate(10)

    # async the laser cmd server with the actuation end using ros param
    rospy.set_param('laser_cmd', 0)

    servo1_pub = rospy.Publisher('/servo1_angle', Float32, queue_size=1)
    servo1_msg = Float32()
    
    servo2_pub = rospy.Publisher('/servo2_angle', Float32, queue_size=1)
    servo2_msg = Float32()

    ar_pose_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, ar_pose_cb, 
                                        [ servo1_pub, servo1_msg, servo2_pub, servo2_msg])

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