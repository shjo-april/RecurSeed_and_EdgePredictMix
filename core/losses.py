# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch
from torch import nn
from torch.nn import functional as F

from tools.ai import torch_utils

class BCE_with_weights(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets, weights=None):
        loss = -(targets * F.logsigmoid(inputs) + (1 - targets) * F.logsigmoid(-inputs))
        if weights is not None:
            loss *= weights
        return loss.mean()

class L1Loss(nn.Module):
    def __init__(self, ignore_index=255, bg_mode='zero', balance=None):
        super().__init__()

        self.ignore_index = ignore_index

        self.bg_mode = bg_mode # zero or ignore
        self.balance = balance # None or fg+bg or fg

        self.to(torch.device('cuda:0'))

    def to(self, device):
        self.device = device
        return self

    def forward(self, preds, gts, labels=None):
        preds = torch_utils.resize(preds, gts.shape[2:])

        if self.balance is None:
            if self.bg_mode == 'zero':
                gts[gts == self.ignore_index] = 0.
            else:
                preds = preds[gts != self.ignore_index]
                gts = gts[gts != self.ignore_index]

            # print(preds.shape, preds.min(), preds.max())
            # print(gts.shape, gts.min(), gts.max())

            loss = torch.abs(preds - gts)
            loss = torch.mean(loss)
        else:
            b, c, h, w = gts.shape

            preds = preds.view(b * c, h * w)
            gts = gts.view(b * c, h * w)
            labels = labels.view(b * c)

            # set fg/bg masks/counts
            fg_counts = (labels > 0).sum()
            bg_counts = (labels == 0).sum()
            
            # preds = preds[gts != self.ignore_index]
            # gts = gts[gts != self.ignore_index]

            if fg_counts > 0:
                fg_preds = preds[labels > 0]
                fg_gts = gts[labels > 0]

                if self.bg_mode == 'zero':
                    fg_gts[fg_gts == self.ignore_index] = 0.
                else:
                    fg_preds = fg_preds[fg_gts != self.ignore_index]
                    fg_gts = fg_gts[fg_gts != self.ignore_index]

                fg_loss = torch.mean(torch.abs(fg_preds - fg_gts))

                # print(fg_gts.shape, fg_gts.min(), fg_gts.max())
            else:
                fg_loss = torch.zeros(1).to(self.device)
            
            if bg_counts > 0:
                bg_preds = preds[labels == 0]
                bg_gts = gts[labels == 0]

                if self.bg_mode == 'zero':
                    bg_gts[bg_gts == self.ignore_index] = 0.
                else:
                    bg_preds = bg_preds[bg_gts != self.ignore_index]
                    bg_gts = bg_gts[bg_gts != self.ignore_index]

                bg_loss = torch.mean(torch.abs(bg_preds - bg_gts))

                # print(bg_gts.shape, bg_gts.min(), bg_gts.max())
            else:
                bg_loss = torch.zeros(1).to(self.device)

            # print(preds.shape, gts.shape, fg_counts, bg_counts)
            
            if self.balance == 'fg':
                loss = fg_loss
            elif self.balance == 'fg+bg':
                loss = (fg_loss + bg_loss) / 2
        
        return loss