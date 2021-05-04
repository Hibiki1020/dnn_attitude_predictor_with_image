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

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import sys

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from common import bnn_network

class BnnAttitudeEstimationWithImage:

    def __init__(self):
        print("BNN attitude estimation with camera image")

        #Parameter of ROS
        self.frame_id = rospy.get_param('~frame_id', '/base_link')
        print("self.frame_id ==> ", self.frame_id)

        #Parameters of BNN
        weights_path = rospy.get_param('~weights_path', '/home/bnn_attitude_predictor_with_image/test2/weights/regression1775train318valid224resize0.5mean0.5stdAdam1e-05lrcnn0.0001lrfc70batch50epoch.pth')
        print("weights_path  ==> ", weights_path)

        resize = rospy.get_param('~resize')
        print("reize         ==> ", resize)

        mean_element = rospy.get_param('~mean_element', 0.5)
        print("mean_element  ==> ", mean_element)

        std_element = rospy.get_param('~std_element', 0.5)
        print("std_element   ==> ", std_element)

        self.num_mcsampling = rospy.get_param('~num_mcsampling', 25)
        print("self.num_mcsampling => ", self.num_mcsampling)

        self.dropout_rate = rospy.get_param('~dropout_rate', 0.1)
        print("self.dropout_rate   => ", self.dropout_rate)

        self.subscribe_topic_name = rospy.get_param('~subscribe_topic_name', '/color_image')

        #ROS subscriber
        self.sub_color_img = rospy.Subscriber( self.subscribe_topic_name, ImageMsg, self.callbackColorImage, queue_size=1, buff_size=2**24)

        #ROS publisher
        self.pub_vector = rospy.Publisher("/bnn/g_vector", Vector3Stamped, queue_size=1)
        self.pub_accel = rospy.Publisher("/bnn/g_vector_with_cov", Imu, queue_size=1)
        self.pub_epistemic = rospy.Publisher("/bnn/epistemic_uncertain", Float32 , queue_size=1)

        #msg
        self.v_msg = Vector3Stamped()
        self.accel_msg = Imu()

        self.epistemic = 0.0 #Initial Value
        
        #BNN internal parameter
        self.expected_value = Vector3Stamped() # expected_value => gravity vector

        #OpenCV
        self.bridge = CvBridge()
        self.color_img_cv = np.empty(0)

        #BNN
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)

        self.img_transform = self.getImageTransform(resize, mean_element, std_element)
        self.net = self.getNetwork(resize, weights_path, self.dropout_rate)
        self.enable_dropout()

    def getImageTransform(self, resize, mean_element, std_element):
        mean = ([mean_element, mean_element, mean_element])
        std = ([std_element, std_element, std_element])

        img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return img_transform

    def getNetwork(self, resize, weights_path, dropout_rate):
        #VGG16を使用した場合
        net = bnn_network.Network(resize, dim_fc_out=3, dropout_rate=dropout_rate, use_pretrained_vgg=False)
        print(net)

        net.to(self.device)
        net.eval() #change inference mode

        #load
        if torch.cuda.is_available():
            loaded_weights = torch.load(weights_path)
            print("GPU  ==>  GPU")
        else:
            loaded_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("GPU  ==>  CPU")
        
        #nn.load_state_dict(loaded_weights)
        net.load_state_dict(loaded_weights)
        return net
    
    def enable_dropout(self):
        #enable dropout when inference
        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    def callbackColorImage(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("msg.encoding = ", msg.encoding)
            print("self.color_img_cv.shape = ", self.color_img_cv.shape)

            print("---------------------")
            start_clock = rospy.get_time()
            list_outputs = self.bnnPrediction()
            output_inference = self.bnnPrediction_Once()
            print("Period [s]: ", rospy.get_time() - start_clock)

            self.expected_value, var_inf = self.calc_excepted_value_and_variance(list_outputs)
            self.epistemic = self.calc_epistemic(output_inference, self.expected_value, var_inf)
            print("Variance : ", var_inf)
            print("Epistemic: ", self.epistemic)
            print("---------------------")

            print("\n")
            print("\n")



            self.InputToMsg(output_inference, list_outputs)
            self.publication(msg.header.stamp)

        except CvBridgeError as e:
            print(e)

    def transformImage(self):
        ## color
        color_img_pil = self.cvToPIL(self.color_img_cv)
        color_img_tensor = self.img_transform(color_img_pil)
        inputs_color = color_img_tensor.unsqueeze_(0)
        inputs_color = inputs_color.to(self.device)
        return inputs_color
    
    def cvToPIL(self, img_cv):
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil


    def calc_excepted_value_and_variance(self, list_outputs):
        mean = np.array(list_outputs).mean(0)
        var = np.var(list_outputs, axis=(0,1))

        return mean, var

    def normalize(self, v):
        l2 = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
        l2[l2==0] = 1
        return v/l2
    
    def calc_epistemic(self, output_inference, expected_value, var_inf):
        #See formulation (4) in Yarin Gal's paper: What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision
        out_inf = np.array(output_inference)
        out_inf = self.normalize(out_inf)

        exp_val = np.array(expected_value)
        exp_val = self.normalize(exp_val)

        print("var_inf = ", var_inf)
        print("out_inf = ",out_inf)
        print("exp_val = ", exp_val)
        relation = np.array(out_inf - exp_val)

        #epistemic = var_inf + output_inference.T * output_inference + expected_value.T * expected_value
        epistemic = var_inf + np.dot(relation.T, relation)
        return epistemic

    def bnnPrediction_Once(self):
        inputs_color = self.transformImage()
        print("inputs_color.size() = ", inputs_color.size())
        output_inf = self.net(inputs_color)
        output = output_inf.cpu().detach().numpy()[0]

        return output

    def bnnPrediction(self):
        ##Inference##
        inputs_color = self.transformImage()
        print("inputs_color.size() = ", inputs_color.size())
        list_outputs = []
        for _ in range(self.num_mcsampling): #MCサンプリングの回数だけ推論する
            outputs = self.net(inputs_color) #do inference
            list_outputs.append(outputs.cpu().detach().numpy()[0])

        return list_outputs

    def getCovMatrix(self, outputs):
        cov = np.cov(outputs, rowvar=False, bias=True)
        return cov

    def InputToMsg(self, output_inference, list_outputs):
        tmp_list_outputs = list_outputs
        tmp_list_outputs.append(output_inference)

        cov = self.getCovMatrix(tmp_list_outputs)
        
        #Vector3Stamped
        self.v_msg.vector.x = -output_inference[0]
        self.v_msg.vector.y = -output_inference[1]
        self.v_msg.vector.z = -output_inference[2]

        self.InputNanToImuMsg(self.accel_msg)
        self.accel_msg.linear_acceleration.x = -output_inference[0]
        self.accel_msg.linear_acceleration.y = -output_inference[1]
        self.accel_msg.linear_acceleration.z = -output_inference[2]
        for i in range(cov.size):
            self.accel_msg.linear_acceleration_covariance[i] = cov[i//3, i%3]

    def InputNanToImuMsg(self, imu):
        imu.orientation.x = math.nan
        imu.orientation.y = math.nan
        imu.orientation.z = math.nan
        imu.orientation.w = math.nan
        imu.angular_velocity.x = math.nan
        imu.angular_velocity.y = math.nan
        imu.angular_velocity.z = math.nan
        imu.linear_acceleration.x = math.nan
        imu.linear_acceleration.y = math.nan
        imu.linear_acceleration.z = math.nan
        for i in range(len(imu.linear_acceleration_covariance)):
            imu.orientation_covariance[i] = math.nan
            imu.angular_velocity_covariance[i] = math.nan
            imu.linear_acceleration_covariance[i] = math.nan
    
    def publication(self, stamp):
        print("delay[s]: ", (rospy.Time.now() - stamp).to_sec())
        
        #vector3stamped
        self.v_msg.header.stamp = stamp
        self.v_msg.header.frame_id = self.frame_id
        self.pub_vector.publish(self.v_msg)

        ## Imu
        self.accel_msg.header.stamp = stamp
        self.accel_msg.header.frame_id = self.frame_id
        self.pub_accel.publish(self.accel_msg)

        #Epistemic
        self.pub_epistemic.publish(self.epistemic)

def main():
    #Set ip ROS node
    rospy.init_node('ros_infer', anonymous=True)

    bnn_attitude_estimation_with_image = BnnAttitudeEstimationWithImage()

    rospy.spin()

if __name__ == '__main__':
    main()