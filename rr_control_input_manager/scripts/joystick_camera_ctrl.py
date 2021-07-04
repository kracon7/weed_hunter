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



def camThread(running, frameBuffer):
    print("camThread pid:", os.getpid(), " tid:", threading.current_thread().ident)
    import pyrealsense2 as rs2
    ctx = rs2.context()
    devs = ctx.query_devices()
    print("query_devices %d" % devs.size())

    pipe = rs2.pipeline()

    pipe.start()
    for i in range(10):
        while frameBuffer.full():
            time.sleep(0.1)
        frame = pipe.wait_for_frames()
        frameBuffer.put(frame.frame_number)
        print("frame_number: %d" % frame.frame_number)
    pipe.stop()
    
    running.put(False)

def record_cam(output_dir):
    print("camThread pid:", os.getpid(), " tid:", threading.current_thread().ident)
    import pyrealsense2 as rs
    from realsense_device_manager import DeviceManager

    # initialize the camera
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    device_manager = DeviceManager(rs.context(), config)

    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
    else:
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    dispose_frames_for_stablisation = 20
    for frame in range(dispose_frames_for_stablisation):
        device_manager.poll_frames()
    
    # set up save dir
    t = dt.datetime.now()
    HOME = str(Path.home())
    save_dir = os.path.join(HOME, output_dir, '%d-%d_%d-%d'%(t.month, t.day, t.hour, t.minute))
    os.system('mkdir -p {}'.format(save_dir))

    rospy.loginfo('Camera process start')

    i = 0
    now = time.time()
    try:
        while True:
            # get fixed camera image data and save to numpy array
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            infrared_frame = aligned_frames.get_infrared_frame()
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data()) * depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            infrared_image = np.asanyarray(infrared_frame.get_data())
            
            elaps = time.time() - now
            print('Captured image %07d, took %f seconds'%(i, elaps))
            now = time.time()
            np.save(os.path.join(save_dir, 'color_%07d.npy'%(i)), color_image)
            np.save(os.path.join(save_dir, 'depth_%07d.npy'%(i)), depth_image)
            np.save(os.path.join(save_dir, 'infrared_%07d.npy'%(i)), infrared_image)

            i += 1
    
    finally:
        print('stopped')
        pipeline.stop()

def xbutton_cb(data, args):
    global cam_process
    output_dir = args
    print(type(output_dir))

    if not cam_process.is_alive():
        cam_process = mp.Process(target=record_cam, args=(output_dir,)) 
        cam_process.start()
    else:
        rospy.loginfo('Camera process is already running')

def ybutton_cb(data):
    global cam_process
    if cam_process.is_alive():
        cam_process.terminate()
        rospy.loginfo('Camera process terminated.')
    else:
        rospy.loginfo('Camera process is already terminated.')
       
def main(args):
    
    rospy.init_node('joystick_camera_control')
    rate=rospy.Rate(10)

    output_dir = args.output_dir
    global cam_process
    cam_process = mp.Process(target=record_cam, args=(output_dir,))
    a_sub = rospy.Subscriber('/joystick/x_button', Bool, xbutton_cb, (output_dir))
    b_sub = rospy.Subscriber('/joystick/y_button', Bool, ybutton_cb)

    
    while not rospy.is_shutdown():
        rate.sleep()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--output_dir', default='jiacheng/data', help='directory to store images')
    args = parser.parse_args()
    
    main(args)