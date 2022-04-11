# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import math
import torch

from torch import nn
from torch.nn import functional as F

from .abc_modules import Guide_For_Model

from .backbones import resnet
from .backbones import resnest
from .backbones import wide_resnet

from core import layers
from tools.ai import torch_utils

class Backbone(nn.Module, Guide_For_Model):
    def __init__(
            self, 
            backbone,
            norm_fn='bn', 
            activation_fn='relu',
            pretrained=True,
            last_stride=2,
            output_stride=16
        ):
        super().__init__()

        if norm_fn == 'bn':
            self.norm_fn = nn.BatchNorm2d
        elif norm_fn == 'gn':
            self.norm_fn = lambda in_channels: nn.GroupNorm(4, in_channels, eps=1e-3)
        elif norm_fn == 'fix':
            self.norm_fn = layers.FixedBatchNorm
        
        if activation_fn == 'relu':
            self.activation_fn = lambda:nn.ReLU(inplace=True)
        
        if 'wide' in backbone:
            self.model = wide_resnet.build_wide_resnet(backbone, last_stride, pretrained)
            self.in_channels = [128, 256, 512, 1024, 4096]

        elif 'resnet' in backbone:
            self.model = resnet.build_resnet(backbone, self.norm_fn, self.activation_fn, last_stride, pretrained, output_stride)
            self.in_channels = [64, 256, 512, 1024, 2048]

        elif 'resnest' in backbone:
            self.model = resnest.build_resnest(backbone, self.norm_fn, last_stride, pretrained)
            self.in_channels = [64, 256, 512, 1024, 2048]

            if not '50' in backbone:
                self.in_channels[0] *= 2
    
    def global_pooling_2d(self, x, keepdims=False):
        x = self.global_pooling_layer(x)
        if not keepdims:
            x = x[:, :, 0, 0]
        return x
    
    def forward_for_backbone(self, x):
        return self.model(x)

class Modified_DeepLabv3_Plus(Backbone):
    def __init__(
            self, 
            backbone, cls_classes=20, seg_classes=21, norm_fn='bn', 
            feature_size=256, output_stride=16, 
            last_stride=1, freeze=False, mode='xavier', high_channels=256
        ):
        super().__init__(backbone, norm_fn=norm_fn, pretrained=True, last_stride=last_stride, output_stride=output_stride)
        
        # for classification
        self.global_pooling_layer = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Conv2d(self.in_channels[-1], cls_classes, kernel_size=1, bias=True)
        self.classifier = nn.Conv2d(self.in_channels[-1], cls_classes, kernel_size=1, bias=False)
        
        # self.initialize([self.classifier])
        
        # for segmentation
        seg_norm_fn = nn.BatchNorm2d
        if freeze:
            seg_norm_fn = lambda in_channels: nn.GroupNorm(4, in_channels, eps=1e-3)

        self.aspp = layers.ASPP(
            self.in_channels[-1], 
            feature_size, 
            output_stride, 
            seg_norm_fn,
            mode
        )
        self.decoder = layers.Decoder(
            self.in_channels[1], 
            feature_size, 
            seg_classes, 
            seg_norm_fn,
            mode,
            high_channels=high_channels
        )

        self.freeze = freeze

    def train(self, mode=True):
        super().train(mode)

        if self.freeze:
            self.model.eval()

            for layer in self.model.modules():
                if isinstance(layer, self.norm_fn):
                    layer.bias.requires_grad = False
                    layer.weight.requires_grad = False
    
    def forward(self, inputs, with_cam=False, with_scm=False, with_segment=False, same=True, interpolation='nearest'):
        output_dict = {}
        
        C1, C2, C3, C4, C5 = self.model(inputs)
        
        if self.freeze:
            C1 = C1.detach()
            C2 = C2.detach()
            C3 = C3.detach()
            C4 = C4.detach()
            C5 = C5.detach()
        
        if with_cam:
            output_dict['features'] = self.classifier(C5)
            output_dict['cls_logits'] = self.global_pooling_2d(output_dict['features'])
        else:
            x = self.global_pooling_2d(C5, keepdims=True)
            output_dict['cls_logits'] = self.classifier(x)[:, :, 0, 0]
        
        if with_scm:
            output_dict['f_dict'] = {
                'C1' : C1,
                'C2' : C2,
                'C3' : C3,
                'C4' : C4,
                'C5' : C5
            }
        
        if with_segment:
            x = self.aspp(C5); D1 = x
            x, D2 = self.decoder(x, C2, with_features=True)

            if with_scm:
                output_dict['f_dict']['D1'] = D1
                output_dict['f_dict']['D2'] = D2

            output_dict['seg_logits'] = x
            if same:
                output_dict['seg_logits'] = torch_utils.resize(output_dict['seg_logits'], inputs.size()[2:], mode=interpolation)
        
        return output_dict
    
    def forward_with_scales(
            self, 
            image, image_size,
            scales=[1.0], feature_names=[],
            hflip=False, dictionary=False,
            
            with_cls=True,
            with_cam=True,
            with_segment=False,
            with_single=False,
            same=True,

            interpolation='nearest'
        ):
        max_h = 0
        max_w = 0

        seg_h = 0
        seg_w = 0
        
        pred_classes = []
        pred_masks = []

        cam_dict = {}
        feature_dict = {}

        images = image.unsqueeze(0)
        B, C, H, W = images.size()

        with_scm = len(feature_names) > 0
        
        if 'D1' in feature_names:
            with_segment = True

        for scale in scales:
            # rescale
            scaled_images = torch_utils.resize(images, image_size, scale=scale)
            if hflip:
                scaled_images = torch.cat([scaled_images, scaled_images.flip(-1)], dim=0)
            
            # inference
            with torch.no_grad():
                # with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = self.forward(scaled_images, with_cam=with_cam, with_scm=True, with_segment=with_segment, same=same, interpolation=interpolation)
            
            # 1. classification
            if with_cls:
                pred_class = torch.sigmoid(output_dict['cls_logits'])

                pred_classes.append(pred_class[0])
                if hflip:
                    pred_classes.append(pred_class[1])
            
            # 2. cam
            if with_cam:
                # update the maximum size
                if max_h == 0 or max_w == 0:
                    h, w = output_dict['f_dict']['C3'].size()[2:]

                    max_h = max(h, max_h)
                    max_w = max(w, max_w)

                # resize feature maps
                cam = torch_utils.resize(output_dict['features'], (max_h, max_w))
                
                # add cams
                cam_dict[scale] = [cam[0]]
                if hflip:
                    cam_dict[scale].append(cam[1].flip(-1))

            # 3. scm
            if with_scm:
                for name in feature_names:
                    output_dict['f_dict'][name] = torch_utils.resize(output_dict['f_dict'][name], (max_h, max_w))
                    if hflip:
                        output_dict['f_dict'][name][1] = output_dict['f_dict'][name][1].flip(-1)
                
                feature_dict[scale] = output_dict['f_dict']

            # 4. segmentation
            if with_segment:
                if seg_h == 0:
                    B, C, seg_h, seg_w = output_dict['seg_logits'].size()

                pred_mask = torch.softmax(output_dict['seg_logits'], dim=1)
                pred_mask = torch_utils.resize(pred_mask, (seg_h, seg_w), mode=interpolation)
                
                pred_masks.append(pred_mask[0])
                if hflip:
                    pred_masks.append(pred_mask[1].flip(-1))

        output_dict = {}

        if with_cls:
            pred_class = torch.mean(torch.stack(pred_classes), dim=0, keepdim=True)
            output_dict['pred_class'] = pred_class
        
        if with_segment:
            if with_single:
                single_mask = torch.mean(torch.stack(pred_masks[:2]), dim=0, keepdim=True)    
                output_dict['single_mask'] = single_mask
            
            output_dict['pred_masks'] = []
            for i, scale in enumerate(scales):
                i *= 2
                pred_mask = torch.mean(torch.stack([pred_masks[i], pred_masks[i+1]]), dim=0, keepdim=True)
                output_dict['pred_masks'].append(pred_mask)
            
            pred_mask = torch.mean(torch.stack(pred_masks), dim=0, keepdim=True)
            output_dict['pred_mask'] = pred_mask
        
        if with_cam:
            if dictionary:
                output_dict['single_cams'] = cam_dict
            else:
                cams = []
                for scale in scales:
                    cams += cam_dict[scale]
                output_dict['single_cams'] = cams
        
        if with_scm:
            if dictionary:
                output_dict['features'] = feature_dict
            else:
                feature_list = []
                    
                for name in feature_names:
                    f = torch.cat([feature_dict[scale][name] for scale in scales], dim=0)
                    f = torch.sum(f, dim=0).unsqueeze(0)
                    
                    feature_list.append(f)

                output_dict['features'] = feature_list
        
        return output_dict
