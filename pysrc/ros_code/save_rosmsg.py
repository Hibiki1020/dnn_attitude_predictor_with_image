#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from cv_bridge import CvBridge, CvBridgeError

import cv2
import PIL.Image as Image
import math
import numpy as np
import argparse
import yaml
import os
import time
import csv

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import sys

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from common import bnn_network

class SaveROSMsg:

    def __init__(self):
        self.frame_id = rospy.get_param('~frame_id', '/base_link')
        
        self.onecam_checker = rospy.get_param('~1cam_checker',"True")
        #If this parameter is false, mode is changed to 4cam
        
        self.front_cam_topic = rospy.get_param('~front_cam_topic', '/camera_f/decompressed_image')
        self.left_cam_topic = rospy.get_param('~left_cam_topic', '/camera_l/decompressed_image')
        self.right_cam_topic = rospy.get_param('~right_cam_topic','/camera_r/decompressed_image')
        self.back_cam_topic = rospy.get_param('~back_cam_topic', '/camera_b/decompressed_image')

        self.velodyne_topic = rospy.get_param('~velodyne_topic', '/velodyne_packets')

        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data')

        self.wait_sec = float(rospy.get_param('~wait_sec', '3'))

        self.dataset_top_path = rospy.get_param('~dataset_top_path','/home/ssd_dir/dataset_image_to_gravity_ozaki/stick/1cam/')
        self.picname = rospy.get_param('~picname', '20210420_143420')
        self.csv_name = rospy.get_param('~csv_name', 'imu_camera.csv')

        self.gvec_min = float(  rospy.get_param('~gvec_min','0.001')  )
        self.gvec_max = float(  rospy.get_param('~gvec_max','100.0')  )

        self.catch_imu_checker = False
        self.catch_img_checker = False

        self.csv_path = os.path.join(self.dataset_top_path, self.csv_name)

        self.picture_counter = 0

        if(self.onecam_checker==True):
            #OpenCV
            self.bridge = CvBridge()
            self.color_img_cv = np.empty(0)
            self.sub_image = rospy.Subscriber(self.front_cam_topic, ImageMsg, self.callbackColorImage, queue_size=1)
        else:
            print("sss")
            quit()
        
        self.imu_data = Imu()
        self.sub_imu_msg = rospy.Subscriber(self.imu_topic, Imu, self.callbackImuMsg, queue_size=1)

    def callbackImuMsg(self, msg):
        self.imu_data = msg
        #print("catch imu data")
        self.catch_imu_checker = True

    def gvec_norm(self, imu_data, gvec_min, gvec_max):
        tmp_csv_data = [imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z]
        array = np.array(tmp_csv_data)
        norm = np.linalg.norm(array, ord=2, axis=-1, keepdims=True)

        print("norm :", norm)

        checker = False

        if norm > gvec_min and norm < gvec_max:
            checker = True
        else:
            checker = False
        
        return checker

    def callbackColorImage(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.catch_img_checker = True
            print("Got Image msg")

            imu_gravity_checker = self.gvec_norm(self.imu_data, self.gvec_min, self.gvec_max)

            if(imu_gravity_checker==True):
                self.save_data()
            
            time.sleep(self.wait_sec) #wait X sec

            self.catch_img_checker = False
            self.catch_imu_checker = False
            self.picture_counter += 1

        except CvBridgeError as e:
            print(e)

    def save_data(self):
        picture_name = self.picname + "picture" + str(self.picture_counter) + ".png"
        
        pic_path = os.path.join(self.dataset_top_path, picture_name)
        csv_path = os.path.join(self.dataset_top_path, self.csv_name)

        with open(csv_path, 'a') as csvfile: # 'a' -> 書き込み用に開き、ファイルが存在する場合は末尾に追記する
            #この列が書き込まれる
            tmp_csv_data = [self.imu_data.linear_acceleration.x, self.imu_data.linear_acceleration.y, self.imu_data.linear_acceleration.z, picture_name]
            cv2.imwrite(pic_path, self.color_img_cv)
            
            writer = csv.writer(csvfile)
            writer.writerow(tmp_csv_data)
        
        csvfile.close()

def main():
    rospy.init_node('save_rosmsg', anonymous=True)

    save_rosmsg = SaveROSMsg()
    rospy.spin()

if __name__ == '__main__':
    main()