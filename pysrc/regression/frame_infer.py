import cv2
import PIL.Image as Image
import math
import numpy as np
import time
import argparse
import yaml
import os
import csv

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms


import sys
sys.path.append('../')
from common import dnn_network

class DnnAttitudeEstimationWithImageFrame:
    def __init__(self, CFG):
        print("DNNAttitudeEstimationWithImageFrame")
        
        self.CFG = CFG
        #contain yaml data to variance
        self.method_name = CFG["method_name"]

        self.dataset_frame_path = CFG["dataset_frame_path"]
        print(self.dataset_frame_path)
        self.csv_name = CFG["csv_name"]
        self.weights_top_path = CFG["weights_top_path"]
        self.weights_file_name = CFG["weights_file_name"]

        self.weights_path = os.path.join(self.weights_top_path, self.weights_file_name)
        
        self.log_file_path = CFG["log_file_path"]
        self.log_file_name = CFG["log_file_name"]

        self.frame_id = CFG["frame_id"]

        self.resize = CFG["resize"]
        self.mean_element = CFG["mean_element"]
        self.std_element = CFG["std_element"]
        self.num_mcsampling = CFG["num_mcsampling"]
        self.dropout_rate = CFG["dropout_rate"]

        #saving parameter in csv file
        self.v_vector = []
        self.accel_msg = []
        self.epistemic = []

        self.expected_value = []

        self.mean_epistemic = []
        self.ave_epistemic = 0.0

        #open_cv
        #self.bridge = CvBridge()
        self.color_img_cv = np.empty(0)

        #DNN
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)

        self.img_transform = self.getImageTransform(self.resize, self.mean_element, self.std_element)
        self.net = self.getNetwork(self.resize, self.weights_path, self.dropout_rate)

    def getNetwork(self, resize, weights_path, dropout_rate):
        #VGG16を使用した場合
        net = dnn_network.Network(resize, dim_fc_out=3, dropout_rate=dropout_rate, use_pretrained_vgg=False)
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
        
        net.load_state_dict(loaded_weights)
        return net

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

    def enable_dropout(self):
        #enable dropout when inference
        can_dropout = False
        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
                can_dropout = True

        return can_dropout
    
    def spin(self):
        data_list = self.get_image_data() #CSVファイル内の画像ファイル名を絶対パスに
        result_csv = self.frame_infer(data_list)
        self.save_csv(result_csv, data_list)

    def normalize(self, v):
        l2 = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
        l2[l2==0] = 1
        return v/l2

    def frame_infer(self, data_list):
        print("Start Inference")

        result_csv = []

        for row in data_list:

            print("---------------------")
            self.color_img_cv = cv2.imread(row[3]) #get image data in bgr6
            inputs_color = self.transformImage()
            print("color_img_cv.shape = ", self.color_img_cv.shape)
            #print("input image shape  = ", inputs_color)
            print("Transform input image")
            print("---------------------")

            start_clock = time.time()
            output_inference_tmp = np.array( self.DNNPrediction() )

            output_inference = self.normalize(output_inference_tmp)
                
            print("Period [s]: ", time.time() - start_clock)
            print("---------------------")

            print("\n")
            print("\n")


            #x, y, z, var, epistemic, image_file_name
            tmp_result = [output_inference[0], output_inference[1], output_inference[2], row[3]]

            result_csv.append(tmp_result)

        return result_csv
    
    def save_csv(self, result_csv, data_list):
        
        result_csv_path = os.path.join(self.log_file_path, self.log_file_name)
        csv_file = open(result_csv_path, 'w')
        csv_w = csv.writer(csv_file)

        for row in result_csv:
            csv_w.writerow(row)

        csv_file.close()

    def DNNPrediction(self):
        inputs_color = self.transformImage()
        print("inputs_color.size() = ", inputs_color.size())
        output_inf = self.net(inputs_color)
        output = output_inf.cpu().detach().numpy()[0]

        return output

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

    def get_image_data(self):
        image_address_list = []
        
        csv_path = os.path.join(self.dataset_frame_path, self.csv_name)

        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row[3] = os.path.join(self.dataset_frame_path, row[3])
                image_address_list.append(row)

        return image_address_list

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("./frame_infer.py")

    parser.add_argument(
        '--frame_infer_config', '-fic',
        type=str,
        required=False,
        default='/home/ros_catkin_ws/src/dnn_attitude_predictor_with_image/config/frame_infer_config.yaml',
        help='Frame infer config yaml file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #load yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.frame_infer_config)
        CFG = yaml.safe_load(open(FLAGS.frame_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.frame_infer_config)
        quit()

    dnn_attitude_predictor_with_image_frame = DnnAttitudeEstimationWithImageFrame(CFG)
    
    #Get image data and do inference
    dnn_attitude_predictor_with_image_frame.spin()