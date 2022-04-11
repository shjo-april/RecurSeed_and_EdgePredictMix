# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch
from torch import nn

from torch.nn import functional as F

from tools.ai import torch_utils

class CAM:
    def __init__(self):
        pass
    
    def get_max_value(self, f):
        return F.adaptive_max_pool2d(f, (1, 1)) + 1e-5
    
    def __call__(self, f):
        islist = isinstance(f, list)

        if islist: 
            f = torch.stack(f)

        f = F.relu(f)

        if islist: 
            f = torch.sum(f, dim=0)
        
        return f / self.get_max_value(f)

class SCG:
    def __init__(
            self, 
            first_th, second_th,
            foreground_th, background_th,
            iteration=1, second_weight=1,
        ):
        self.first_th = first_th
        self.second_th = second_th

        self.foreground_th = foreground_th
        self.background_th = background_th
        
        self.iteration = iteration
        self.second_weight = second_weight

        self.device = torch.device('cuda:0')

    def to(self, device):
        self.device = device
    
    def normalize(self, value, mode='plus', dim=1, keepdim=True, eps=1e-10):
        if mode == 'plus':
            return value / (torch.sum(value, dim=dim, keepdim=True) + eps)
        elif mode == 'minmax':
            if keepdim:
                min_value = torch.min(value, dim=dim, keepdim=True)[0]
                max_value = torch.max(value, dim=dim, keepdim=True)[0]
                return (value - min_value) / (max_value - min_value + eps)
            else:
                return (value - value.min()) / (value.max() - value.min() + eps)

    def get_HSC(self, SC1, SC2):
        return torch.max(SC1, self.second_weight * SC2)

    def generate_SC(self, features):
        b, c, h, w = features.size()
        
        # B, C, H, W -> B, H, W, C
        features = features.permute(0, 2, 3, 1).contiguous()
        # B, H, W, C -> B, H * W, C
        features = features.view(b, h * w, c)

        # frobenius normalization
        norm_f = features / (torch.norm(features, dim=2, keepdim=True) + 1e-10)

        # first order
        SC = F.relu(torch.matmul(norm_f, norm_f.transpose(1, 2)))
        # print(SC.min(), SC.max(), SC.size())

        if self.first_th > 0:
            SC[SC < self.first_th] = 0

        SC1 = self.normalize(SC)

        # second order
        SC[:, torch.arange(h * w), torch.arange(w * h)] = 0
        SC = SC / (torch.sum(SC, dim=1, keepdim=True) + 1e-10)

        base_th = 1 / (h * w)
        second_th = base_th * self.second_th
        
        SC2 = SC.clone()
        
        for _ in range(self.iteration):
            SC2 = torch.matmul(SC2, SC)
            SC2 = self.normalize(SC2)
        
        # print(SC2.min(), SC2.max(), SC2.size(), second_th)

        if self.second_th > 0:
            SC2[SC2 < second_th] = 0
        
        return SC1, SC2

    def generate_HSC(self, fs):
        HSC_list = []
        
        for f in fs:
            SC1, SC2 = self.generate_SC(f)

            HSC = self.get_HSC(SC1, SC2)
            HSC = self.normalize(HSC)

            HSC_list.append(HSC)

        HSC = torch.sum(torch.stack(HSC_list), dim=0)
        return HSC

    def generate_SCM(self, cams, HSC):
        batch_size, classes, h, w = cams.size()
        ids = torch.arange(h * w)

        flatted_cams = cams.view(batch_size, classes, h * w)

        scms = []
        for b in range(batch_size):
            scm_per_batch = []
            for c in range(classes):
                # flatted_cam = cams[b][c].view(-1)
                flatted_cam = flatted_cams[b, c]
                
                fg_ids = ids[flatted_cam >= self.foreground_th]
                if len(fg_ids) > 0:
                    fg_HSC = self.normalize(HSC[b, :, fg_ids], 'minmax', dim=0)

                    fg_HSC = torch.sum(fg_HSC, dim=1).view((h, w))
                    fg_HSC = self.normalize(fg_HSC, 'minmax', keepdim=False)
                else:
                    fg_HSC = torch.zeros(h, w).to(self.device)
                
                bg_ids = ids[flatted_cam <= self.background_th]
                if len(bg_ids) > 0:
                    bg_HSC = self.normalize(HSC[b, :, bg_ids], 'minmax', dim=0)

                    bg_HSC = torch.sum(bg_HSC, dim=1).view((h, w))
                    bg_HSC = self.normalize(bg_HSC, 'minmax', keepdim=False)
                else:
                    bg_HSC = torch.zeros(h, w).to(self.device)
                
                scm = F.relu(fg_HSC - bg_HSC)

                scm = self.normalize(scm, 'minmax', keepdim=False)
                scm_per_batch.append(scm)
            
            scm = torch.stack(scm_per_batch)
            scms.append(scm)

        return torch.stack(scms)
    
    def get_preds_using_ms(self, cams, fs):
        return self.__call__(cams, fs)

    def get_preds_using_hflip(self, cams, fs, size=None):
        fs = [(f[0] + f[1].flip(-1)).unsqueeze(0) for f in fs]
        if size is not None:
            fs = [torch_utils.resize(f, size) for f in fs]
        
        return self.__call__(cams, fs)

    def __call__(self, cams, fs):
        without_batch = len(cams.size()) == 3
        if without_batch:
            cams = cams.unsqueeze(0)
            fs = [f.unsqueeze(0) for f in fs]

        # 1. calulate HSC from feature maps.
        HSC = self.generate_HSC(fs)

        # 2. calculate SCM
        SCM = self.generate_SCM(cams, HSC)

        if without_batch:
            SCM = SCM[0]

        return SCM
