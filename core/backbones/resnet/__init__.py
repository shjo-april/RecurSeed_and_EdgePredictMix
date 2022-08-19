# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch.utils.model_zoo as model_zoo

from .model import ResNet
from .model import Bottleneck
from .model import urls_dic, layers_dic

def build_resnet(model_name, norm_fn, activation_fn, last_stride=2, pretrained=True, output_stride=16):
    if output_stride == 16:
        strides=(2, 2, 2, last_stride)
        dilations=(1, 1, 1, 1)

    elif output_stride == 8:
        strides=(2, 2, 1, last_stride)
        dilations=(1, 1, 2, 4)
    
    model = ResNet(
        Bottleneck, 
        layers_dic[model_name], 
        strides=strides, dilations=dilations,
        batch_norm_fn=norm_fn, activation_fn=activation_fn
    )
    
    if pretrained:
        state_dict = model_zoo.load_url(urls_dic[model_name])
        model.load_state_dict(state_dict, strict=False)
    
    return model
