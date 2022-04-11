# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch.utils.model_zoo as model_zoo

from .model import resnest50, resnest101, resnest200, resnest269

def build_resnest(model_name, norm_fn, last_stride=2, pretrained=True):
    if last_stride == 2:
        dilation, dilated = 1, False
    else:
        dilation, dilated = 2, False
    
    model = eval(model_name)(pretrained=pretrained, dilated=dilated, dilation=dilation, norm_layer=norm_fn)
    
    return model