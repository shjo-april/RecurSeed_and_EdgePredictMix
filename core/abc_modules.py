# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import cv2
import math

from PIL import Image

from torch import nn
from torch.nn import functional as F

from abc import ABC

from tools.general import xml_utils
from tools.general import json_utils

class Guide_For_Model(ABC):
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def initialize(self, weights, conv_init='kaiming'):
        for m in weights:
            if isinstance(m, nn.Conv2d):
                if conv_init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                elif conv_init == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
                    
            # scracthed weights
            else:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
        return groups

class Guide_For_Dataset:
    def __init__(self, root_dir, domain, flags, json_path):
        self.image_dir = root_dir + f'{domain}/image/'
        self.xml_dir = root_dir + f'{domain}/xml/'
        self.mask_dir = root_dir + f'{domain}/mask/'
        
        self.image_names = os.listdir(self.image_dir)
        
        self.flags = flags
        self.func_dict = {
            'image_id' : self.get_id,
            'image' : self.get_image,
            
            'bbox' : self.get_bbox,
            'mask' : self.get_mask,

            'tag' : self.get_tag
        }

        self.data_dict = json_utils.read_json(json_path)
        
        self.class_dict = self.data_dict['class_dict']
        self.num_classes = self.data_dict['num_classes']
    
    def __len__(self):
        return len(self.image_names)
    
    def get_id(self, image_id):
        return image_id

    def get_image(self, image_id):
        image_path = self.image_dir + image_id + '.jpg'
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = None
        return image
    
    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            image = Image.open(mask_path)
        else:
            image = None
        return image
    
    def get_bbox(self, image_id):
        xml_path = self.xml_dir + image_id + '.xml'
        if os.path.isfile(xml_path):
            bboxes, classes = xml_utils.read_xml(xml_path, keep_difficult=True)
        else:
            bboxes, classes = [], []
        return bboxes, classes
    
    def get_tag(self, image_id):
        _, classes = self.get_bbox(image_id)
        return list({name:0 for name in classes}.keys())
    
    def __getitem__(self, index):
        image_id = self.image_names[index].replace('.jpg', '')
        return {flag:self.func_dict[flag](image_id) for flag in self.flags}

    def get_data_from_image_id(self, image_id):
        return {flag:self.func_dict[flag](image_id) for flag in self.flags}
