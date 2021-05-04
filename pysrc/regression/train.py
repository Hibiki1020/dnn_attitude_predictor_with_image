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
from common import bnn_network


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default='/home/ros_catkin_ws/src/bnn_attitude_predictor_with_image/config/train_config.yaml',
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
    #net = bnn_network.Network(resize, list_dim_fc_out=[100, 18, 3], dropout_rate=0.1, use_pretrained_vgg=True)
    net = bnn_network.Network(resize, dim_fc_out=3, dropout_rate=0.1, use_pretrained_vgg=True)


    ##Criterion
    criterion = nn.MSELoss()

    #train
    trainer = trainer_mod.Trainer(
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
        log_path,
        graph_path
    )

    trainer.train()
