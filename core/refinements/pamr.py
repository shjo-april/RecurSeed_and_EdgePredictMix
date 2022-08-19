import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

class LocalAffinity(nn.Module):
    def __init__(self, dilations=[1]):
        super().__init__()

        self.dilations = dilations
        self.kernel = self._init_aff()
    
    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1

        return weight

    def forward(self, x):
        self.kernel = self.kernel.type_as(x)

        B,K,H,W = x.size()
        x = x.view(B*K,1,H,W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d]*4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B,K,-1,H,W)

class LocalIdentity(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        return weight

class LocalStd(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1

        return weight
    
    def forward(self, x):
        x = super().forward(x)
        x = x.std(dim=2, keepdim=True)
        return x

class LocalAffinityAbs(LocalAffinity):
    def forward(self, x):
        x = super().forward(x)
        x = torch.abs(x)
        return x

class PAMR(nn.Module):
    def __init__(self, num_iter=1, dilations=[1], sigma=0.1):
        super().__init__()

        self.num_iter = num_iter
        self.sigma = sigma
        self.eps = 1e-8 # to avoid saturation

        self.aff_abs = LocalAffinityAbs(dilations)
        self.aff_std = LocalStd(dilations)
        self.aff_ide = LocalIdentity(dilations)

    def forward(self, x, m):
        distance = self.aff_abs(x)
        std = self.aff_std(x)

        # distance.size() = torch.Size([1, 3, 48, 256, 383])
        # std.size() = torch.Size([1, 3, 1, 256, 383])

        # print(distance.size())
        # print(std.size())
        
        aff = -distance / (self.sigma * std + self.eps)
        aff = torch.mean(aff, dim=1, keepdim=True)
        aff = torch.softmax(aff, dim=2)

        # aff.size() = torch.Size([1, 1, 48, 256, 383])
        # mask.size() = torch.Size([1, K, 256, 383])

        # print(aff.size())
        # print(mask.size())

        for _ in range(self.num_iter):
            # m.size() = torch.Size([1, K, 48, 256, 383])
            m = self.aff_ide(m)
            m = (aff * m).sum(dim=2)
        
        return m
