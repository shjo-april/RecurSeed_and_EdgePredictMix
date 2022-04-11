# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import math
import torch

from torch import nn
from torch.nn import functional as F

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_fn=None, bias=False, mode='kaiming'):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.bn = norm_fn(planes)
        self.relu = nn.ReLU(inplace=True)

        self.initialize([self.atrous_conv, self.bn], mode)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def initialize(self, modules, mode='kaiming'):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if mode == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif mode == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride, norm_fn, mode='kaiming'):
        super().__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        
        self.aspp1 = ASPPModule(in_channels, out_channels, 1, padding=0, dilation=dilations[0], norm_fn=norm_fn, mode=mode)
        self.aspp2 = ASPPModule(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], norm_fn=norm_fn, mode=mode)
        self.aspp3 = ASPPModule(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], norm_fn=norm_fn, mode=mode)
        self.aspp4 = ASPPModule(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], norm_fn=norm_fn, mode=mode)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            norm_fn(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.initialize([self.conv1, self.bn1] + list(self.global_avg_pool.modules()), mode)
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

    def initialize(self, modules, mode='kaiming'):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if mode == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif mode == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, norm_fn, mode='kaiming', high_channels=256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, high_channels, 1, bias=False)
        self.bn1 = norm_fn(high_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(out_channels + high_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # nn.Conv2d(out_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
            nn.Conv2d(out_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        )
        
        self.initialize([self.conv1, self.bn1] + list(self.classifier.modules()), mode)
    
    def forward(self, x, x_low_level, with_features=False):
        x_low_level = self.conv1(x_low_level)
        x_low_level = self.bn1(x_low_level)
        x_low_level = self.relu(x_low_level)

        x = F.interpolate(x, size=x_low_level.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, x_low_level), dim=1); f = x
        x = self.classifier(x)

        if with_features:
            return x, f
        else:
            return x

    def initialize(self, modules, mode='kaiming'):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if mode == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif mode == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP_For_DeepLabv2(nn.Module):
    def __init__(self, in_channels, num_classes, mode='kaiming'):
        super().__init__()
        
        dilations = [6, 12, 18, 24]
        
        self.aspp1 = nn.Conv2d(
            in_channels, num_classes, 
            kernel_size=3, stride=1, 
            padding=dilations[0], dilation=dilations[0], bias=True)

        self.aspp2 = nn.Conv2d(
            in_channels, num_classes, 
            kernel_size=3, stride=1, 
            padding=dilations[1], dilation=dilations[1], bias=True)

        self.aspp3 = nn.Conv2d(
            in_channels, num_classes, 
            kernel_size=3, stride=1, 
            padding=dilations[2], dilation=dilations[2], bias=True)
        
        self.aspp4 = nn.Conv2d(
            in_channels, num_classes, 
            kernel_size=3, stride=1, 
            padding=dilations[3], dilation=dilations[3], bias=True)

        self.initialize([self.aspp1, self.aspp2, self.aspp3, self.aspp4], mode)
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        
        x = x1 + x2 + x3 + x4
        return x

    def initialize(self, modules, mode='kaiming'):
        prior = 0.01
        
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if mode == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif mode == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight)
                    # torch.nn.init.xavier_uniform_(m.bias)
                elif mode == 'retina':
                    m.weight.data.fill_(0)
                    m.bias.data.fill_(-math.log((1.0 - prior) / prior))
