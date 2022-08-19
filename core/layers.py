# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import math
import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _quadruple

from tools.ai import torch_utils

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding=0, dilation=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, low_channels=48):
        super().__init__()
        
        self.low_block = ConvBlock(in_channels, low_channels, 1)
        
        self.classifier = nn.Sequential(
            ConvBlock(out_channels + low_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(0.5),

            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(0.1),

            nn.Conv2d(out_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def forward(self, x, x_low):
        x_low = self.low_block(x_low)

        x = torch_utils.resize(x, size=x_low.size()[2:])
        x = torch.cat((x, x_low), dim=1)

        x = self.classifier(x)

        return x

class ASPP_For_DeepLabv3(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super().__init__()
        
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        
        self.aspp1 = ConvBlock(in_channels, out_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ConvBlock(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ConvBlock(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ConvBlock(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(in_channels, out_channels, 1)
        )
        
        self.block = nn.Sequential(
            ConvBlock(out_channels * 5, out_channels, 1),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = torch_utils.resize(x5, size=x4.size()[2:])

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.block(x)

        return x

class ASPP_For_DeepLabv2(nn.Module):
    def __init__(self, in_channels, num_classes):
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

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        
        x = x1 + x2 + x3 + x4
        return x
