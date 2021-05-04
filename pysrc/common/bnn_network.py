from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, dim_fc_out, dropout_rate, use_pretrained_vgg=True):
        super(Network, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained_vgg)
        self.cnn_feature = vgg.features

        dim_fc_in = 512*(resize//32)*(resize//32)
        self.fc = nn.Sequential(
            #Layer1
            nn.Linear(dim_fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            #layer2
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            #Layer3
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            #Layer4
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            #Layer5
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            #Final Layer
            nn.Linear(16, dim_fc_out)
        )

    def getParamValueList(self):
        list_cnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                # print("cnn: ", param_name)
                list_cnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_cnn_param_value: ",list_cnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_cnn_param_value, list_fc_param_value

    def forward(self, x):

        x = self.cnn_feature(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        l2norm = torch.norm(x[:, :3].clone(), p=2, dim=1, keepdim=True)
        x[:, :3] = torch.div(x[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return x