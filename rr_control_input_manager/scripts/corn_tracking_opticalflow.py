#!/home/jungseok/venvs/pytorch/bin/python3.6

# Modified https://github.com/yashs97/object_tracker/blob/master/multi_label_tracking.py

import numpy as np
import argparse
import cv2 
import time
from imutils.video import FPS 
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision

import random

import operator

from utils.cv_bridge import image_to_numpy, numpy_to_image
import rospy
from std_msgs.msg import String, Int32, Float32, Bool, Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
import message_filters

CLASS_NAMES = ["__background__", "corn_stem"]
# Labels of Network.
labels = { 0: 'background', 1: 'corn'}
lk_params = dict(winSize = (50,50), maxLevel = 4, 
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
total_frames = 0
tracking_started = False
frame_count = 50
centroids = np.zeros([1, 1, 2], dtype=np.float32) #?
corn_dict = dict()
color_dict = dict()
corn_id_bbox = []


class CornTrackingOF(object):
    def __init__(self):
        self.loop_rate = rospy.Rate(100)


        # color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)

        sub = message_filters.Subscriber("/d435/color/image_raw", Image)
        self.cache = message_filters.Cache(sub, 2, allow_headerless=True)
        self.cache.registerCallback(self.callback)


        # color_sub = rospy.Subscriber('/d435/color/image_raw', Image, self.callback)
        self.corn_pub = rospy.Publisher("predictions", String, queue_size=2)
        self.detect_res = rospy.Publisher("detections", Image, queue_size=2)
        self.im_w = 848
        self.im_h = 480

        rospy.loginfo('Loading network module...')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.load("./faster-rcnn-corn_bgr8_ep100.pt")

        rospy.loginfo("Initialization Done!")


    def callback(self, cache):

        global total_frames
        global centroids
        global corn_dict
        global color_dict
        global tracking_started
        global previous
        global current
        global corn_id_bbox
        # print(self.cache.getOldestTime())
        previous = self.cache.getElemBeforeTime(self.cache.getOldestTime()) #get the oldest message
        current= self.cache.getElemAfterTime(self.cache.getLastestTime()) #get the newest message
        #If you have a non stamped message, add the stamp as a global variable
        global previous_stamp
        global current_stamp
        previous_stamp = self.cache.getOldestTime()
        current_stamp= self.cache.getLastestTime()



        # Solve all of perception here...
        # rospy.loginfo('Received frame at time: %d'%(current_stamp))

        prev_frame = image_to_numpy(previous)
        frame = image_to_numpy(current)
        assert (frame.shape[0] == self.im_h) and (frame.shape[1] == self.im_w), \
                'Image size incorrect, expected %d, %d but got %d, %d instead'%(self.im_h, self.im_w, frame.shape[0], np.color.shape[1])


        # run network on rgb image to predict vanishing point and corn lines


        # if frame is None: #end of video file
        #     break
        # running the object detector every nth frame 
        if total_frames % int(frame_count)-1 == 0:
        
            pred_boxes, pred_class, pred_score = self.get_prediction(frame, 0.5)
            centroids = np.zeros([1, 1, 2], dtype=np.float32)

            # only if there are predictions
            if pred_boxes != None:
                corn_dict = dict()
                for i in range(len(pred_boxes)):
                    corn_dict[i]=dict()
                corn_dict['centroids']=dict()

                for i in range(len(pred_boxes)):
                # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
                    color = list(np.random.random(size=3) * 256)
                    # print("i color", i, color)
                    tracking_id = int(i)
                    confidence = pred_score[i]

                    xLeftBottom = int(pred_boxes[i][0][0]) 
                    yLeftBottom = int(pred_boxes[i][0][1])
                    xRightTop   = int(pred_boxes[i][1][0])
                    yRightTop   = int(pred_boxes[i][1][1])

                    # print class and confidence          
                    label = pred_class[i] +": "+ str(confidence)             
                    # print(label)

                    x = (xLeftBottom + xRightTop)/2
                    y = (yLeftBottom + yRightTop)/2

                    corn_dict[i]['bbox'] = [(xLeftBottom,yLeftBottom),(xRightTop,yRightTop)]
                    corn_dict[i]['centroid'] =[(x,y)]
                    corn_dict['centroids'][tuple((x,y))]=[]

                    # bbox_dict[tuple((x,y))]=[(xLeftBottom,yLeftBottom),(xRightTop,yRightTop)]
                    # print("bbox_dict", bbox_dict)
                    frame = cv2.rectangle(frame,(xLeftBottom,yLeftBottom),(xRightTop,yRightTop), color, thickness=2) ### added today
                    # draw the centroid on the frame
                    frame = cv2.circle(frame, (int(x),int(y)), 15, color, -1)
                    print("before if STATE i %d frame %d x y: %d %d" % (i, total_frames, x, y))
                    tracking_started = True
                    if i == 0:
                        color_dict = dict()
                        centroids[0,0,0] = x
                        centroids[0,0,1] = y
                        color_dict[tuple(color)]=[(x,y)]

                    else:
                        centroid = np.array([[[x,y]]],dtype=np.float32)
                        centroids = np.append(centroids,centroid,axis = 0)
                        color_dict[tuple(color)]=[(x,y)]

            original_centroids = centroids ########
            # else:
            #     color_dict=dict()

        else:   # track an object only if it has been detected
            if centroids.sum() != 0 and tracking_started:
                next1, st, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                                centroids, None, **lk_params)

                good_new = next1[st==1]
                good_old = centroids[st==1]


                # print("color dict", color_dict)
                old_centroids = centroids
                corn_id_bbox = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    # Returns a contiguous flattened array as (x, y) coordinates for new point
                    a, b = new.ravel()
                    c, d = old.ravel()
                    distance = np.sqrt((a-c)**2 + (b-d)**2)
                    # distance between new and old points should be less than
                    # 200 for 2 points to be same the object
                    if distance < 200 :
                        corn_dict['centroids'][corn_dict[i]['centroid'][0]].append((a,b))
                        for color, centroids_list in color_dict.items():
                            # print("centroid list", centroids_list)
                            for centroids in centroids_list:
                                if centroids==(c,d):
                                    color_dict[color].append((a,b))
                                    color_old = color
                                    frame = cv2.circle(frame, (a, b), 15, color_old, -1)
                        # for centroids, bbox in bbox_dict.items():
                        #     if centroids==(c,d):
                        #         bbox_coor = bbox
                        
                        #### how to contorl id?
                        res = tuple(map(operator.sub, (c,d),corn_dict[i]['centroid'][0]))
                        new_bbox_coor1 = tuple(map(operator.add, corn_dict[i]['bbox'][0], res))
                        new_bbox_coor2 = tuple(map(operator.add, corn_dict[i]['bbox'][1], res))
                        new_bbox_coor1 = tuple(map(int, new_bbox_coor1))
                        new_bbox_coor2 = tuple(map(int, new_bbox_coor2))

                        frame = cv2.rectangle(frame, new_bbox_coor1, new_bbox_coor2, color_old, thickness=2) ### added today                    
                        frame = cv2.putText(frame, str(i), new_bbox_coor2,cv2.FONT_HERSHEY_SIMPLEX, 1, color_old, 5, cv2.LINE_AA)
                        # frame = cv2.rectangle(frame, bbox_coor[0], bbox_coor[1], color_old, thickness=2) ### added today
                        # frame = cv2.circle(frame, (a, b), 15, color_old, -1)
                        frame = cv2.putText(frame, str(total_frames), (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 5, cv2.LINE_AA)
                        # corn_id_bbox[i]=[new_bbox_coor1, new_bbox_coor2]
                        corn_id_bbox.append([new_bbox_coor1, new_bbox_coor2])
                print("total fr", total_frames,"corn_id", corn_id_bbox)
                text = "hello"
                self.corn_pub.publish(text.encode("utf-8"))

                centroids = good_new.reshape(-1, 1, 2)


        # print(corn_dict)
        # break

        total_frames += 1







        # now = time.time()
        # pred_boxes, pred_class, pred_score = self.get_prediction(frame, 0.5)


        # print("Took %.5fsec to predict "%(
                # time.time()-now), pred_class, pred_score)

        self.detect_res.publish(numpy_to_image(frame, "bgr8"))
        self.corn_pub.publish([corn_dict])




    def get_prediction(self, img, confidence=0.5):
        """
        get_prediction
        parameters:
            - img_path - path of the input image
            - confidence - threshold value for prediction score
        method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - class, box coordinates are obtained, but only prediction score > threshold
            are chosen.
        
        """
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model = torch.load("./output/faster-rcnn-corn_bgr8_ep100.pt")
        # img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img).to(self.device)
        pred = self.model([img])
        pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        if len([x for x in pred_score if x>confidence])!=0:
            pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]
            pred_score = pred_score[:pred_t+1]
        else:
            pred_boxes, pred_class, pred_score = None, None, None

        return pred_boxes, pred_class, pred_score


    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    # parser.add_argument('--output_dir', default='jiacheng/data', help='directory to store images')
    # parser.add_argument('--cam_type', default='shepherd', type=str, help='type of camera setup, dual or single')
    # args, unknown = parser.parse_known_args()

    rospy.init_node("corn_detection_node", anonymous=True)
    corn_detection_node = CornTrackingOF()
    corn_detection_node.start()

