
import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def extract_chunks(file_in, output_dir, chunks):
    bagfile = rosbag.Bag(file_in)
    messages = bagfile.get_message_count()
    m_per_chunk = int(round(float(messages) / float(chunks)))
    chunk = 0
    m = 0
    outbag = rosbag.Bag(os.path.join(output_dir, "chunk_%04d.bag" % chunk), 'w')
    for topic, msg, t in bagfile.read_messages():
        m += 1
        if m % m_per_chunk == 0:
            outbag.close()
            chunk += 1
            outbag = rosbag.Bag("chunk_%04d.bag" % chunk, 'w')
        outbag.write(topic, msg, t)
    outbag.close()


def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--chunks", default=2, type=int, help="number of chunks to split")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print("Split bagfile %s into %d chunks"%(args.bag_file, args.chunks))

    bag = extract_chunks(args.bag_file, args.output_dir, args.chunks)
    

if __name__ == '__main__':
    main()