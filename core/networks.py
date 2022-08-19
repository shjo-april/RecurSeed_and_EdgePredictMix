# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import copy
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from typing import List
from .abc_modules import Guide_For_Model

from .backbones import resnet

from .layers import ASPP_For_DeepLabv2, ASPP_For_DeepLabv3, Decoder, FixedBatchNorm

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
        elif norm_fn == 'fix':
            self.norm_fn = FixedBatchNorm
        
        if activation_fn == 'relu':
            self.activation_fn = lambda:nn.ReLU(inplace=True)
        
        if 'resnet' in backbone:
            self.model = resnet.build_resnet(backbone, self.norm_fn, self.activation_fn, last_stride, pretrained, output_stride)
            self.in_channels = [64, 256, 512, 1024, 2048]

class RSEPM(Backbone):
    def __init__(self, 
            backbone, num_classes=20, 
            last_stride=1, class_fn='sigmoid',
            output_stride=16, feature_size=256
        ):
        super().__init__(backbone, norm_fn='bn', pretrained=True, last_stride=last_stride, output_stride=output_stride)
        
        self.class_fn = class_fn
        self.num_classes = num_classes
        
        self.global_pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(self.in_channels[-1], self.num_classes, kernel_size=1, bias=False)

        self.aspp = ASPP_For_DeepLabv3(
            self.in_channels[-1], 
            feature_size, 
            output_stride
        )
        self.decoder = Decoder(
            self.in_channels[1], 
            feature_size, 
            num_classes+1, # with background
            feature_size
        )
    
    def global_pooling_2d(self, x, keepdims=False):
        x = self.global_pooling_layer(x)
        if not keepdims:
            x = x[:, :, 0, 0]
        return x
    
    def forward(self, x, same=False, with_decoder=False, with_features=False, interpolation='bilinear'):
        output_dict = {}

        C1, C2, C3, C4, C5 = self.model(x)
        
        output_dict['cams'] = self.classifier(C5)
        output_dict['cls_logits'] = self.global_pooling_2d(output_dict['cams'])

        if with_decoder:
            x_aspp = self.aspp(C5)

            output_dict['seg_logits'] = self.decoder(x_aspp, C2)
            if same:
                output_dict['seg_logits'] = torch_utils.resize(output_dict['seg_logits'], x.size()[2:], mode=interpolation)
        
        if with_features:
            output_dict['f_dict'] = {
                'C1' : C1,
                'C2' : C2,
                'C3' : C3,
                'C4' : C4,
                'C5' : C5
            }
        
        return output_dict

    def forward_with_scales(
            self, 
            image, image_size,
            scales=[1.0], hflip=False, with_decoder=False, with_cam=True,
            same=False, interpolation='bilinear', cls_postprocess='mean',

            feature_names=[], dictionary=False # SCG
        ):
        max_h = 0
        max_w = 0

        seg_h = 0
        seg_w = 0

        pred_cams = []
        pred_masks = []
        pred_classes = []
        feature_dict = {}

        images = image.unsqueeze(0)

        for scale in scales:
            # rescale
            scaled_images = torch_utils.resize(images, image_size, scale=scale)
            if hflip: scaled_images = torch.cat([scaled_images, scaled_images.flip(-1)], dim=0)
            
            # inference
            with torch.no_grad():
                output_dict = self.forward(scaled_images, same=same, with_decoder=with_decoder, with_features=True, interpolation=interpolation)
            
            # update the maximum size
            if max_h == 0 or max_w == 0:
                h, w = output_dict['f_dict']['C3'].size()[2:]
                
                max_h = max(h, max_h)
                max_w = max(w, max_w)
            
            # resize feature maps
            if with_cam:
                cam = torch_utils.resize(output_dict['cams'], (max_h, max_w), mode=interpolation)

                pred_cams.append(cam[0])
                if hflip: pred_cams.append(cam[1].flip(-1))

            # add logits
            if self.class_fn == 'sigmoid': pred_class = torch.sigmoid(output_dict['cls_logits'])
            else: pred_class = torch.softmax(output_dict['cls_logits'], dim=1)
            
            pred_classes.append(pred_class[0])
            if hflip: pred_classes.append(pred_class[1])

            # for SCG
            for name in feature_names:
                output_dict['f_dict'][name] = torch_utils.resize(output_dict['f_dict'][name], (max_h, max_w), mode=interpolation)
                if hflip: output_dict['f_dict'][name][1] = output_dict['f_dict'][name][1].flip(-1)
            
            feature_dict[scale] = output_dict['f_dict']

            # for segmentation
            if with_decoder:
                if seg_h == 0 or seg_w == 0:
                    B, C, seg_h, seg_w = output_dict['seg_logits'].size()

                pred_mask = torch.softmax(output_dict['seg_logits'], dim=1)
                pred_mask = torch_utils.resize(pred_mask, (seg_h, seg_w), mode=interpolation)
                
                pred_masks.append(pred_mask[0])
                if hflip:
                    pred_masks.append(pred_mask[1].flip(-1))
        
        output_dict = {
            'pred_cams': pred_cams,
        }

        if cls_postprocess == 'mean':
            output_dict['pred_class'] = torch.mean(torch.stack(pred_classes), dim=0)
        else:
            output_dict['pred_class'] = torch.max(torch.stack(pred_classes), dim=0)[0]

        if with_decoder:
            output_dict['pred_mask'] = torch.mean(torch.stack(pred_masks), dim=0)

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

class DeepLabv3_Plus(Backbone):
    def __init__(self, backbone, num_classes=21, output_stride=16, feature_size=256, low_channels=48):
        super().__init__(backbone, norm_fn='bn', pretrained=True, last_stride=1, output_stride=output_stride)

        self.aspp = ASPP_For_DeepLabv3(
            self.in_channels[-1], 
            feature_size, 
            output_stride
        )
        self.decoder = Decoder(
            self.in_channels[1], 
            feature_size, 
            num_classes,
            low_channels=low_channels
        )

    def forward(self, inputs, same=True, interpolation='bilinear'):
        output_dict = {}

        _, C2, _, _, C5 = self.model(inputs)

        x = self.aspp(C5)
        x = self.decoder(x, C2)

        output_dict['seg_logits'] = x
        if same:
            output_dict['seg_logits'] = torch_utils.resize(x, inputs.size()[2:], mode=interpolation)
        
        return output_dict

    def forward_with_scales(
            self, 
            image, image_size,
            scales=[1.0], hflip=False,
            interpolation='bilinear'
        ):
        seg_h = 0
        seg_w = 0
        
        pred_masks = []

        images = image.unsqueeze(0)

        for scale in scales:
            # rescale
            scaled_images = torch_utils.resize(images, image_size, scale=scale)
            if hflip: scaled_images = torch.cat([scaled_images, scaled_images.flip(-1)], dim=0)
            
            # inference
            with torch.no_grad():
                output_dict = self.forward(scaled_images, interpolation=interpolation)
            
            # postprocessing
            if seg_h == 0:
                B, C, seg_h, seg_w = output_dict['seg_logits'].size()

            pred_mask = torch.softmax(output_dict['seg_logits'], dim=1)
            pred_mask = torch_utils.resize(pred_mask, (seg_h, seg_w), mode=interpolation)
            
            pred_masks.append(pred_mask[0])
            if hflip: pred_masks.append(pred_mask[1].flip(-1))

        output_dict = {}
        
        pred_mask = torch.mean(torch.stack(pred_masks), dim=0, keepdim=True)
        output_dict['pred_mask'] = pred_mask

        return output_dict

class DeepLabv2(Backbone):
    def __init__(self, backbone, num_classes=21, output_stride=8):
        super().__init__(backbone, norm_fn='bn', pretrained=True, last_stride=1, output_stride=output_stride)

        self.aspp = ASPP_For_DeepLabv2(self.in_channels[-1], num_classes)

    def forward(self, inputs, same=True, interpolation='bilinear'):
        output_dict = {}

        C1, C2, C3, C4, C5 = self.model(inputs)

        x = self.aspp(C5)

        output_dict['seg_logits'] = x
        if same:
            output_dict['seg_logits'] = torch_utils.resize(output_dict['seg_logits'], inputs.size()[2:], mode=interpolation)
        
        return output_dict

    def forward_with_scales(
            self, 
            image, image_size,
            scales=[1.0], hflip=False,
            interpolation='bilinear'
        ):
        seg_h = 0
        seg_w = 0
        
        pred_masks = []

        images = image.unsqueeze(0)
        
        for scale in scales:
            # rescale
            scaled_images = torch_utils.resize(images, image_size, scale=scale)
            if hflip: scaled_images = torch.cat([scaled_images, scaled_images.flip(-1)], dim=0)
            
            # inference
            with torch.no_grad():
                output_dict = self.forward(scaled_images, interpolation=interpolation)
            
            # postprocessing
            if seg_h == 0:
                B, C, seg_h, seg_w = output_dict['seg_logits'].size()

            pred_mask = torch.softmax(output_dict['seg_logits'], dim=1)
            pred_mask = torch_utils.resize(pred_mask, (seg_h, seg_w), mode=interpolation)
            
            pred_masks.append(pred_mask[0])
            if hflip: pred_masks.append(pred_mask[1].flip(-1))
        
        output_dict = {}
        
        pred_mask = torch.mean(torch.stack(pred_masks), dim=0, keepdim=True)
        output_dict['pred_mask'] = pred_mask
        
        return output_dict

class MeanShift(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, input):
        if self.training:
            return input
        return input - self.running_mean.view(1, 2, 1, 1)

class AffinityNet(Backbone):
    def __init__(
            self, 
            backbone, 
            path_index=None, 
            feature_size=32, last_stride=1,
            backbone_norm_fn='fix', aff_norm_fn='gn',
            with_instance=False
        ):
        super().__init__(backbone, norm_fn=backbone_norm_fn, pretrained=True, last_stride=last_stride)

        self.backbone_norm_fn = backbone_norm_fn

        if aff_norm_fn == 'gn':
            norm_fn = lambda in_channels: nn.GroupNorm(4, in_channels, eps=1e-3)
            act_fn = lambda in_channels: nn.ReLU(inplace=True)
        elif aff_norm_fn == 'bn':
            norm_fn = nn.BatchNorm2d
            act_fn = lambda in_channels: nn.ReLU(inplace=True)
        
        # for edge
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(self.in_channels[0], feature_size, 1, bias=False),
            norm_fn(feature_size),
            act_fn(feature_size),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(self.in_channels[1], feature_size, 1, bias=False),
            norm_fn(feature_size),
            act_fn(feature_size),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(self.in_channels[2], feature_size, 1, bias=False),
            norm_fn(feature_size),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            act_fn(feature_size),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(self.in_channels[3], feature_size, 1, bias=False),
            norm_fn(feature_size),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            act_fn(feature_size),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(self.in_channels[4], feature_size, 1, bias=False),
            norm_fn(feature_size),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            act_fn(feature_size),
        )
        self.fc_edge6 = nn.Conv2d(feature_size * 5, 1, 1, bias=True)
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

        if path_index is not None:
            self.path_index = path_index
            self.n_path_lengths = len(self.path_index.path_indices)
            for i, pi in enumerate(self.path_index.path_indices):
                self.register_buffer("path_indices_" + str(i), torch.from_numpy(pi))

        self.with_instance = with_instance
        if self.with_instance:
            self.mean_shift = MeanShift(2)

            self.fc_dp1 = nn.Sequential(
                nn.Conv2d(self.in_channels[0], 64, 1, bias=False),
                norm_fn(64),
                act_fn(64),
            )
            self.fc_dp2 = nn.Sequential(
                nn.Conv2d(self.in_channels[1], 128, 1, bias=False),
                norm_fn(128),
                act_fn(128),
            )
            self.fc_dp3 = nn.Sequential(
                nn.Conv2d(self.in_channels[2], 256, 1, bias=False),
                norm_fn(256),
                act_fn(256),
            )
            self.fc_dp4 = nn.Sequential(
                nn.Conv2d(self.in_channels[3], 256, 1, bias=False),
                norm_fn(256),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                act_fn(256),
            )
            self.fc_dp5 = nn.Sequential(
                nn.Conv2d(self.in_channels[4], 256, 1, bias=False),
                norm_fn(256),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                act_fn(256),
            )

            self.fc_dp6 = nn.Sequential(
                nn.Conv2d(256*3, 256, 1, bias=False),
                norm_fn(256),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                act_fn(256),
            )
            self.fc_dp7 = nn.Sequential(
                nn.Conv2d(448, 256, 1, bias=False),
                norm_fn(256),
                act_fn(256),
                nn.Conv2d(256, 2, 1, bias=False),
                self.mean_shift,
            )

            if path_index is not None:
                self.register_buffer(
                    'disp_target',
                    torch.unsqueeze(torch.unsqueeze(torch.from_numpy(path_index.search_dst).transpose(1, 0), 0), -1).float())

    def train(self, mode=True):
        super().train(mode)

        if self.backbone_norm_fn == 'fix':
            self.model.eval()

    def get_edge(self, x, image_size=512, stride=4):
        x = x.unsqueeze(0)
        x = torch.cat([x, x.flip(-1)], dim=0)
        
        feat_size = (x.size(2)-1)//stride+1, (x.size(3)-1)//stride+1
        x = F.pad(x, [0, image_size-x.size(3), 0, image_size-x.size(2)])

        edge_out = self.forward(x)
        if self.with_instance:
            edge_out = edge_out[0]

        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        
        return edge_out

    def forward(self, x, training=False):
        xs = self.model(x)

        if self.backbone_norm_fn == 'fix':
            x1 = xs[0].detach()
            x2 = xs[1].detach()
            x3 = xs[2].detach()
            x4 = xs[3].detach()
            x5 = xs[4].detach()
        else:
            x1 = xs[0]
            x2 = xs[1]
            x3 = xs[2]
            x4 = xs[3]
            x5 = xs[4]
        
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)
        edge4 = self.fc_edge4(x4)
        edge5 = self.fc_edge5(x5)

        edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        if self.with_instance:
            dp1 = self.fc_dp1(x1)
            dp2 = self.fc_dp2(x2)
            dp3 = self.fc_dp3(x3)
            dp4 = self.fc_dp4(x4)
            dp5 = self.fc_dp5(x5)

            dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))
            dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

            if training: return self.to_affinity(torch.sigmoid(edge)), self.to_pair_displacement(dp_out)
            else: return edge, dp_out
        else:
            if training: return self.to_affinity(torch.sigmoid(edge))
            else: return edge
    
    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)
        
        for i in range(self.n_path_lengths):
            ind = self._buffers["path_indices_" + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)
        return aff_cat

    def to_pair_displacement(self, disp):
        height, width = disp.size(2), disp.size(3)
        radius_floor = self.path_index.radius_floor

        cropped_height = height - radius_floor
        cropped_width = width - 2 * radius_floor

        disp_src = disp[:, :, :cropped_height, radius_floor:radius_floor + cropped_width]

        disp_dst = [disp[:, :, dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
                       for dy, dx in self.path_index.search_dst]
        disp_dst = torch.stack(disp_dst, 2)

        pair_disp = torch.unsqueeze(disp_src, 2) - disp_dst
        pair_disp = pair_disp.view(pair_disp.size(0), pair_disp.size(1), pair_disp.size(2), -1)

        return pair_disp