#!/home/jc/Envs/py36/bin/python3.6


import os
import sys
import argparse

import cv2

import rosbag
import numpy as np
from sensor_msgs.msg import Image


name_to_dtypes = {
    "rgb8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "rgb16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgr8":    (np.uint8,  3),
    "bgra8":   (np.uint8,  4),
    "bgr16":   (np.uint16, 3),
    "bgra16":  (np.uint16, 4),
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),
    
    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8":  (np.uint8,  1),
    "bayer_bggr8":  (np.uint8,  1),
    "bayer_gbrg8":  (np.uint8,  1),
    "bayer_grbg8":  (np.uint8,  1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1":    (np.uint8,   1),
    "8UC2":    (np.uint8,   2),
    "8UC3":    (np.uint8,   3),
    "8UC4":    (np.uint8,   4),
    "8SC1":    (np.int8,    1),
    "8SC2":    (np.int8,    2),
    "8SC3":    (np.int8,    3),
    "8SC4":    (np.int8,    4),
    "16UC1":   (np.uint16,   1),
    "16UC2":   (np.uint16,   2),
    "16UC3":   (np.uint16,   3),
    "16UC4":   (np.uint16,   4),
    "16SC1":   (np.int16,  1),
    "16SC2":   (np.int16,  2),
    "16SC3":   (np.int16,  3),
    "16SC4":   (np.int16,  4),
    "32SC1":   (np.int32,   1),
    "32SC2":   (np.int32,   2),
    "32SC3":   (np.int32,   3),
    "32SC4":   (np.int32,   4),
    "32FC1":   (np.float32, 1),
    "32FC2":   (np.float32, 2),
    "32FC3":   (np.float32, 3),
    "32FC4":   (np.float32, 4),
    "64FC1":   (np.float64, 1),
    "64FC2":   (np.float64, 2),
    "64FC3":   (np.float64, 3),
    "64FC4":   (np.float64, 4)
}


def image_to_numpy(msg):
    if not msg.encoding in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
    
    dtype_class, channels = name_to_dtypes[msg.encoding]
    dtype = np.dtype(dtype_class)
    print(msg.encoding)
    dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
    shape = (msg.height, msg.width, channels)

    data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
    data.strides = (
        msg.step,
        dtype.itemsize * channels,
        dtype.itemsize
    )

    if channels == 1:
        data = data[...,0]
    return data


# def main(args):

#     if args.msg_type == 'color':
#         # create output dirs
#         color_dir = os.path.join(args.output_dir, 'color')
#         os.system('mkdir -p {}'.format(color_dir))
#     elif args.msg_type == 'depth':
#         depth_dir = os.path.join(args.output_dir, 'depth')
#         os.system('mkdir -p {}'.format(depth_dir))
    
#     # plt.ion()
#     # fig, ax = plt.subplots(2,1)
    
#     rospy.init_node('iamge_collection')
#     rospy.loginfo("testing virtual env")
#     rate=rospy.Rate(10)

#     count = 0
#     while True:
#         # get robot image data from ROS and convert to numpy array
#         if args.msg_type == 'color':
#             color_data = rospy.wait_for_message('/camera/color/image_raw', Image)
#             if color_data.data:
#                 color_image = image_to_numpy(color_data)
#                 save_jpg(color_image, os.path.join(color_dir, 'color_%d.jpg'%(count)))

#         elif args.msg_type == 'depth':
#             depth_data = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
        
#             if depth_data.data:
#                 depth_image = image_to_numpy(depth_data)
#                 # np.save(os.path.join(depth_dir, 'depth_%d.npy'%(count)), depth_image)
#                 # cv2.imwrite(os.path.join(depth_dir, 'depth_%d.png'%(count)), depth_image)
#                 save_jpg(depth_image, os.path.join(depth_dir, 'depth_%d.tif'%(count)))

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--image_topic", help="Image topic.")
    parser.add_argument("--max_num", default=-1, type=int, help='Maximum number of messages to extract')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = image_to_numpy(msg)
        # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png"%count), cv_img)
        print("Wrote image %i" % count)

        count += 1

    bag.close()


if __name__ == '__main__':
    main()