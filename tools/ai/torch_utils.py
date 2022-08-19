# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import math

import torch
import random
import numpy as np

from torch import nn
from torch.nn import functional as F

from copy import deepcopy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize(x, eps=1e-5):
    C, H, W = x.size()
    max_x = x.view(C, (H * W)).max(dim=1)[0].view(C, 1, 1)
    return x / (max_x + eps)

def resize(tensors, size=None, scale=1.0, mode='bilinear', align_corners=False):
    without_batch = len(tensors.size()) == 3
    if without_batch:
        tensors = tensors.unsqueeze(0)

    if size is None:
        size = tensors.size()[2:]
    
    size = list(size)
    size[0] = int(size[0] * scale)
    size[1] = int(size[1] * scale)

    if mode == 'nearest':
        align_corners = None
    
    _, _, h, w = tensors.size()
    if size[0] != h or size[1] != w:
        tensors = F.interpolate(tensors, size, mode=mode, align_corners=align_corners)
    
    if without_batch:
        tensors = tensors[0]
    
    return tensors

def one_hot_embedding(label, classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (int or list) class labels.
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    
    vector = np.zeros((classes), dtype = np.float32)
    vector[label] = 1.
    return vector

def calculate_parameters(model):
    return sum(param.numel() for param in model.parameters())/1000000.0

def get_numpy(tensor):
    return tensor.cpu().detach().numpy()

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def load_model(model, model_path, strict=True, map_location='cpu'):
    if is_parallel(model):
        model = de_parallel(model)
    model.load_state_dict(torch.load(model_path, map_location=map_location), strict=strict)

def save_model(model, model_path):
    if is_parallel(model):
        model = de_parallel(model)

    torch.save(model.state_dict(), model_path)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model)
        self.ema.eval()  # FP32 EMA

        # print(self.ema, is_parallel(model))

        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
    
    def get_model(self):
        return self.ema
    
    def to(self, device):
        self.ema.to(device)
