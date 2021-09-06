#!/home/jc/Envs/py36/bin/python3.6

import os
import sys
import argparse
import rospy
import struct
import time
import cv2
from pathlib import Path
import numpy as np
import datetime as dt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import tf2_ros
from std_msgs.msg import Int32, Float32, Bool, Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from rr_control_input_manager.msg import CornPlane
import message_filters

import torch
import torchvision.transforms as T
from utils.cv_bridge import image_to_numpy

class Frame():
    """sync-ed frame for side and front view"""
    def __init__(self, front_color, front_depth, side_color, stamp, pose):
        self.front_color = front_color
        self.front_depth = front_depth
        self.side_color = side_color
        self.stamp = stamp
        self.pose = pose


class FitGround(object):
    def __init__(self, args):
        self.loop_rate = rospy.Rate(100)

        front_color_sub = message_filters.Subscriber('/front_d435/color/image_raw', Image)
        front_depth_sub = message_filters.Subscriber('/front_d435/aligned_depth_to_color/image_raw', Image)
        side_color_sub = message_filters.Subscriber('/d435/color/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([
                            front_color_sub, front_depth_sub, side_color_sub], 1, .1)
        ts.registerCallback(self.callback)

        # publish corn plane message
        self.plane_pub = rospy.Publisher("corn_plane", CornPlane, queue_size=1)
        self.plane_msg = CornPlane()

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.im_w = 848
        self.im_h = 480
        self.K = np.array([[615.311279296875,   0.0,             430.1778869628906],
                           [  0.0,            615.4699096679688, 240.68307495117188],
                           [  0.0,              0.0,               1.0]])
        # compute rays in advance
        x, y = np.arange(self.im_w), np.arange(self.im_h)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=2).reshape(-1,2)
        self.rays = np.dot(np.insert(points, 2, 1, axis=1), np.linalg.inv(self.K).T).reshape(self.im_h, self.im_w, 3)
        rospy.loginfo('Ground fit initialization finished...')

        rospy.loginfo('Loading network module...')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.load(os.path.join(str(Path.home()), args.model_path))
        self.model.to(self.device)
        self.CLASS_NAMES = ["__background__", "corn_stem"]

        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()

        # store previous side view frames
        self.frame_buffer = []
        rospy.loginfo('Initialization done!')

    def callback(self, front_color, front_depth, side_color):
        # Solve all of perception here...
        frame_id = side_color.header.frame_id
        t_sec = side_color.header.stamp.secs
        t_nsec = side_color.header.stamp.nsecs
        rospy.loginfo('Received frame with stamp: %d.%d at time %f'%(t_sec, t_nsec, rospy.get_time()))

        np_front_color = image_to_numpy(front_color)
        np_front_depth = image_to_numpy(front_depth).astype('float32') * 1e-3
        np_side_color = image_to_numpy(side_color)
        assert (np_front_color.shape[0] == self.im_h) and (np_front_color.shape[1] == self.im_w), \
                'Image size incorrect, expected %d, %d but got %d, %d instead'%(
                self.im_h, self.im_w, np_color.shape[0], np.color.shape[1])

        transform = None
        try:
            transform = self.tfBuffer.lookup_transform(frame_id, 'map',
                                     side_color.header.stamp)

        except Exception as e:
            print(e)
        
        if transform:   
            rospy.loginfo('got new transformation')
            pose = np.array([transform.transform.translation.x,
                             transform.transform.translation.y,
                             transform.transform.translation.z,
                             transform.transform.rotation.w,
                             transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z])
            curr_frame = Frame(np_front_color, np_front_depth, np_side_color, 
                               side_color.header.stamp, pose)

            self.frame_buffer.append(curr_frame)
            rospy.loginfo('New frame added to frame buffer')

            # remove old frames from the buffer
            dx = np.abs(self.frame_buffer[0].pose[0] - pose[0])
            while dx > 0.08:
                self.frame_buffer.pop(0)
                dx = np.abs(self.frame_buffer[0].pose[0] - pose[0])

            if dx > 0.04:

                rospy.loginfo('estimating plane distance')
                frame1 = self.frame_buffer[0]
                frame2 = self.frame_buffer[-1]
                rel_trans = self.get_rel_trans(frame1.pose, frame2.pose)

                # estimate plane distance
                bbox1, pred_cls_1, pred_score_1 = self.get_bbox(frame1)
                bbox2, pred_cls_2, pred_score_2 = self.get_bbox(frame2)

                rospy.loginfo('faster rcnn done')
                mask1 = self.bbox_to_mask(bbox1, 480, 848)
                mask2 = self.bbox_to_mask(bbox2, 480, 848)
                kp1, des1 = self.sift.detectAndCompute(frame1.side_color, mask1)
                kp2, des2 = self.sift.detectAndCompute(frame2.side_color, mask2)

                rospy.loginfo('sift computation done')
                matches = self.bf_matcher.knnMatch(des1,des2,k=2)
                # Apply ratio test
                good = []
                for m,n in matches:
                    if m.distance < 0.6*n.distance:
                        good.append([m])
                rospy.loginfo('feature matching done')

    def get_rel_trans(self, pose_1, pose_2):
        '''
        Compute the relative transformation between frame1 and frame2
        Input
            frame1 -- dictionary object, stores front rgbd, side color, absolute transformation
        Output
            T -- transformation matrix from frame2 to frame1
        '''
        p1, q1 = pose_1[:3], pose_1[3:]
        p2, q2 = pose_2[:3], pose_2[3:]
        R1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
        R2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]]).as_matrix()
        
        T_map_1, T_2_map = np.eye(4), np.eye(4)
        T_map_1[:3,:3], T_map_1[:3,3] = R1.T, -R1.T @ p1
        T_2_map[:3,:3], T_2_map[:3,3] = R2, p2
        T = T_2_map @ T_map_1
        return T

    def get_bbox(self, frame, confidence=0.8):
        '''
        Get the bounding box for side view corn detection
        Input
            model -- pytorch model object
            frame -- dictionary object, stores front rgbd, side color, absolute transformation
        Output
            bbox -- list object, bounding box position and sise
        '''
        transform = T.Compose([T.ToTensor()])
        img = transform(frame.side_color).to(self.device)
        pred = self.model([img])
        pred_class = [self.CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())

        if len([x for x in pred_score if x>confidence])!=0:
            pred_t = [pred_score.index(s) for s, c in zip(pred_score, pred_class) 
                        if s>confidence and c=='corn_stem'][-1]
            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]
            pred_score = pred_score[:pred_t+1]
        else:
            pred_boxes, pred_class, pred_score = None, None, None

        return pred_boxes, pred_class, pred_score

    def bbox_to_mask(self, bbox, im_h, im_w):
        '''
        generate binary mask according to bounding boxes for feature detection
        '''
        mask = np.zeros((im_h, im_w), dtype='uint8')
        for box in bbox:
            top, bottom, left, right = int(box[0][1]), int(box[1][1]), int(box[0][0]), int(box[1][0])
            mask[top:bottom, left:right] = 1
        return mask

    def publish_plane(self, stamp, d, np):
        '''
        Publish the corn plane in front camera frame
        '''
        # TODO check unlikely cases of the plane model

        # write corn plane message and publish
        self.plane_msg.header.stamp = stamp
        self.plane_msg.d = d
        self.plane_msg.np.x = np[0] 
        self.plane_msg.np.y = np[1]
        self.plane_msg.np.z = np[2]
        self.plane_pub.publish(self.plane_msg)
    
    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--model_path', default='jiacheng/phase2/plane_distance/model/faster-rcnn-corn_bgr8_ep100.pt', 
                                        help='directory to load faster RCNN model')
    args, unknown = parser.parse_known_args()

    rospy.init_node("fit_ground", anonymous=True)
    fit_node = FitGround(args)
    fit_node.start()


