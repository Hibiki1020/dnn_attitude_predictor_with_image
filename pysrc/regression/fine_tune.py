from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import dnn_network

#Fine Tune Class
class FineTuner(trainer_mod.Trainer):
    def __init__(self, 
        method_name,
        train_dataset,
        valid_dataset,
        net,
        criterion,
        optimizer_name,
        lr_cnn,
        lr_fc,
        batch_size,
        num_epochs,
        weights_path,
        pretrained_model_path,
        log_path,
        graph_path):

        self.weights_path = weights_path
        self.log_path = log_path
        self.graph_path = graph_path

        self.setRandomCondition()
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        print("Fine Tine Device: ", self.device)
        self.dataloaders_dict = self.getDataloader(train_dataset, valid_dataset, batch_size)
        self.net = self.getSetNetwork(net, pretrained_model_path)
        self.criterion = criterion
        self.optimizer = self.getOptimizer(optimizer_name, lr_cnn, lr_fc)
        self.num_epochs = num_epochs
        self.str_hyperparameter = self.getStrHyperparameter(method_name, train_dataset, optimizer_name, lr_cnn, lr_fc, batch_size)


    def getSetNetwork(self, net, pretrained_model_path):
        print(net)
        net.to(self.device)

        #load
        if torch.cuda.is_available():
            loaded_weights = torch.load(pretrained_model_path)
            print("Loaded [GPU -> GPU]: ", pretrained_model_path)
        else:
            loaded_weights = torch.load(pretrained_model_path, map_location={"cuda:0": "cpu"})
            print("Loaded [GPU -> GPU]: ", pretrained_model_path)
        
        net.load_state_dict(loaded_weights)

        return net
    
    def getStrHyperparameter(self, method_name, dataset, optimizer_name, lr_cnn, lr_fc, batch_size):
        str_hyperparameter = method_name \
            + str(len(self.dataloaders_dict["train"].dataset)) + "train" \
            + str(len(self.dataloaders_dict["valid"].dataset)) + "valid" \
            + str(dataset.transform.resize) + "resize" \
            + str(dataset.transform.mean[0]) + "mean" \
            + str(dataset.transform.std[0]) + "std" \
            + optimizer_name \
            + str(lr_cnn) + "lrcnn" \
            + str(lr_fc) + "lrfc" \
            + str(batch_size) + "batch" \
            + str(self.num_epochs) + "epoch"
        print("str_hyperparameter = ", str_hyperparameter)
        return str_hyperparameter

if __name__ == '__main__':
    parser = argparse.ArgumentParser('./fine_tune.py')

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default='/home/ros_catkin_ws/src/dnn_attitude_predictor_with_image/config/fine_tune_config.yaml',
        help='Train hyperparameter config file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #load yaml file
    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    #get file paths
    method_name = CFG["method_name"]
    dataset_top_path = CFG["dataset_top_path"]
    experiment_type = CFG["experiment_type"]
    image_env = CFG["image_env"]
    train_sequences = CFG["train"] #string
    valid_sequences = CFG["valid"]
    csv_name = CFG["csv_name"]
    weights_path = CFG["weights_path"]
    pretrained_model_path = CFG["pretrained_model_path"]
    log_path = CFG["log_path"]
    graph_path = CFG["graph_path"]

    #get train and valid root path
    list_train_rootpaths = []
    list_valid_rootpaths = []

    for i in train_sequences:
        tmp_path = dataset_top_path + experiment_type + image_env + i
        list_train_rootpaths.append(tmp_path)
    
    for i in valid_sequences:
        tmp_path = dataset_top_path + experiment_type + image_env + i
        list_valid_rootpaths.append(tmp_path)

    #get hyperparameter for learning
    resize = CFG["hyperparameter"]["resize"]
    mean_element = CFG["hyperparameter"]["mean_element"]
    std_element = CFG["hyperparameter"]["std_element"]
    hor_fov_deg = CFG["hyperparameter"]["hor_fov_deg"]
    optimizer_name = CFG["hyperparameter"]["optimizer_name"]
    lr_cnn = float(CFG["hyperparameter"]["lr_cnn"])
    lr_fc = float(CFG["hyperparameter"]["lr_fc"])
    batch_size = CFG["hyperparameter"]["batch_size"]
    num_epochs = CFG["hyperparameter"]["num_epochs"]

    try:
        print("Copy files to %s for further reference." % log_path)
        copyfile(FLAGS.train_cfg, log_path + "/train_config.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting....")
        quit()

    ##Get train and valid dataset
    train_dataset = dataset_mod.Originaldataset(
        data_list = make_datalist_mod.makeMultiDataList(list_train_rootpaths, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element]),
            hor_fov_deg = hor_fov_deg
        ),
        phase = "train"
    )

    valid_dataset = dataset_mod.Originaldataset(
        data_list = make_datalist_mod.makeMultiDataList(list_valid_rootpaths, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element]),
            hor_fov_deg = hor_fov_deg
        ),
        phase = "valid"
    )

    ##Network
    #use_pretrained_vgg -> False
    net = dnn_network.Network(resize, dim_fc_out=3, dropout_rate=0.1, use_pretrained_vgg=False)


    ##Criterion
    criterion = nn.MSELoss()

    #train
    fine_tune = FineTuner(
        method_name,
        train_dataset,
        valid_dataset,
        net,
        criterion,
        optimizer_name,
        lr_cnn,
        lr_fc,
        batch_size,
        num_epochs,
        weights_path,
        pretrained_model_path,
        log_path,
        graph_path
    )

    fine_tune.train()