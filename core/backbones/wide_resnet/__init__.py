# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch
from .model import Wide_ResNet

def build_wide_resnet(model_name, last_stride=1, pretrained=True):
    model = Wide_ResNet(last_stride)
    
    if pretrained:
        state_dict = torch.load('./pretrained_weights/{}.pth'.format(model_name))
        model.load_state_dict(state_dict) # , strict=False
    
    return model