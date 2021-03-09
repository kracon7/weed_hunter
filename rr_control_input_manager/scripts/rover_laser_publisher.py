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
from math import sin, cos, pi

plt.ion()

def process_frame(color, depth):
    # filter_low = np.array([80, 0,170])
    # filter_high = np.array([120, 40,220])
    filter_low = np.array([80, 120, 0])
    filter_high = np.array([140, 180, 80])
    mask = np.sum(color>=filter_low, axis=2) + np.sum(color<=filter_high, axis=2)
    pixel = np.average(np.argwhere(mask == 6), axis=0)
    print('target pixel position at: {}'.format(pixel))
    try:
        z = depth[np.floor(pixel[0]), np.floor(pixel[1])]
    except Exception as e:
        z = 0.1    

    k_intrisic_matrix=np.array([[918.2091437945788, 0., 639.193207006314],
                                [0., 921.9954810539982,  358.461790471607],
                                [0.,    0.,    1.]])
    ray = np.dot(np.linalg.inv(k_intrisic_matrix), np.array([[pixel[1]],[pixel[0]],[1.]]))

    target_position = z * ray

    return target_position.reshape(-1), mask==6

def compute_laser_cmd(target_position):
    origin = np.array([-92.2359, -21.7008, -90.8480])
    x_axis = np.array([0.0130, -0.0908, 0.9958])
    y_axis = np.array([-0.0059, 0.9958, 0.0909])
    z_axis = np.array([-0.9999, -0.0071, 0.0124])

    vec = origin - target_position
    rad = np.arccos(np.dot(vec, x_axis) / np.linalg.norm(vec))
    angle = 180 - rad/pi*180
    return angle
       
def main(args):
    '''
    First,
    Initialize the Gen3 cartesian space controler
    '''
    rospy.init_node('velocity_controller')
    rate=rospy.Rate(10)
    laser_pub = rospy.Publisher('/laser_cmd', Int32, queue_size=1)
    servo_pub = rospy.Publisher('/servo_angle', Float32, queue_size=1)
    laser_msg = Int32()
    servo_msg = Float32()

    time_start = rospy.Time.now()

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    time_start = time.time()

    data_dir = os.path.join(args.output_dir, args.theta)
    os.system('mkdir -p ' + data_dir)

    if args.visualize:
        fig, ax = plt.subplots(2,1)

    index = 0

    while not rospy.is_shutdown():
        # get image and compute target position
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if args.save_img:
            index += 1
            color_filename = os.path.join(data_dir, 'color_{}.png'.format(index))
            plt.imsave(color_filename, color_image, vmin=0, vmax=255)

        target_position, mask = process_frame(color_image, depth_image)
        print('Found target position in camera frame: {}'.format(target_position))
        angle = compute_laser_cmd(target_position)
        print('Laser angle command: {}'.format(angle))

        if angle > 0. and angle < 20.:
            servo_msg.data = angle
            servo_pub.publish(servo_msg)
            laser_msg.data = 1
            laser_pub.publish(laser_msg)


        # index += 1
        # rospy.loginfo()

        if args.visualize:
            ax[0].imshow(color_image)
            ax[1].imshow(mask)
            plt.pause(0.01)


        # twist_pub.publish(twist_msg)
        rate.sleep()

    pipeline.stop()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--timeout', default=40, type=float, help='total time of image streaming')
    parser.add_argument('--output_dir', default='data', help='directory to store images')
    parser.add_argument('--theta', type=str, default='0', help='angle of servo')
    parser.add_argument('--visualize', type=int, default=0, help='visualize the frames')
    parser.add_argument('--save_img', type=int, default=0, help='save the frames')
    args = parser.parse_args()
    
    main(args)