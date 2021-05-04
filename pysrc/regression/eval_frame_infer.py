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
from common import bnn_network

class FrameInferEval:
    def __init__(self,CFG):
        print("Eval Frame Infer")

        self.frame_infer_log_top_path = CFG["frame_infer_log_top_path"]
        self.frame_infer_log_file_name = CFG["frame_infer_log_file_name"]

        self.dataset_data_top_path = CFG["dataset_data_top_path"]
        self.dataset_data_file_name = CFG["dataset_data_file_name"]

        self.saved_log_csv_top_path = CFG["saved_log_csv_top_path"]
        self.saved_log_csv_file_name = CFG["saved_log_csv_file_name"]

        self.loop_period = CFG["loop_period"]

        self.bookmark_list = []

        self.do_eval()
        self.save_result_csv()

    def save_result_csv(self):
        result_csv_path = os.path.join(self.saved_log_csv_top_path, self.saved_log_csv_file_name)
        csv_file = open(result_csv_path, 'w')
        csv_w = csv.writer(csv_file)

        for row in self.bookmark_list:
            csv_w.writerow(row)
        
        csv_file.close()

    def do_eval(self):
        log_path = os.path.join(self.frame_infer_log_top_path, self.frame_infer_log_file_name)
        dataset_path = os.path.join(self.dataset_data_top_path, self.dataset_data_file_name)

        log_list = []
        with open(log_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                log_list.append(row)

        dataset_list = []
        with open(dataset_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                dataset_list.append(row)

        loop_bar = zip(log_list, dataset_list)
        
        for row_log, row_dataset in loop_bar:
            #pic_path = os.path.join(dataset_data_top_path, row_log[5])
            log_pic = cv2.imread(row_log[5])
            
            log_x = float(row_log[0])
            log_y = float(row_log[1])
            log_z = float(row_log[2])
            log_var = row_log[3]
            log_epistemic = row_log[4]

            data_x = float(row_dataset[0])/9.8
            data_y = float(row_dataset[1])/9.8
            data_z = float(row_dataset[2])/9.8

            print(log_x, log_y, log_z)
            print(data_x, data_y, data_z)

            print("\n")

            diff_x = abs(float(log_x) - float(data_x))
            diff_y = abs(float(log_y) - float(data_y))
            diff_z = abs(float(log_z) - float(data_z))

            tmp_bookmark_list = [row_log[5], log_x, log_y, log_z, diff_x, diff_y, diff_z, log_var, log_epistemic]

            print("diff_x   : ", diff_x)
            print("diff_y   : ", diff_y)
            print("diff_z   : ", diff_z)
            print("epistemic: ",log_epistemic)
            print("Variance : ", log_var)
            
            print("Do you want to save this picture's data? answer in y/n .")
            print("If you want to exit, press q key")

            cv2.imshow('image_log',log_pic)
            answer = cv2.waitKey(0)
                
            if answer == ord('y'):
                self.bookmark_list.append(tmp_bookmark_list)
                print("Save picture and data\n")
            elif answer == ord('q'):
                print("Stop evaluation")
                cv2.destroyAllWindows()
                break
            else:
                print("\n")

            cv2.destroyAllWindows()
            print("\n")







if __name__ == '__main__':

    parser = argparse.ArgumentParser("./eval_frame_infer.py")

    parser.add_argument(
        '--eval_frame_infer_config', '-efic',
        type=str,
        required=False,
        default='/home/ros_catkin_ws/src/bnn_attitude_predictor_with_image/config/eval_frame_infer_config.yaml',
        help='Eval frame infer config yaml file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #load yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.eval_frame_infer_config)
        CFG = yaml.safe_load(open(FLAGS.eval_frame_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.eval_frame_infer_config)
        quit()

    frame_infer_eval = FrameInferEval(CFG)