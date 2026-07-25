# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import cv2
import torch
import numpy as np

from torch.nn import functional as F

from tools.ai import torch_utils
from tools.general import cv_utils

def refine_label_using_edge(seed: torch.Tensor, pseudo_label: torch.Tensor):
    device = seed.get_device()

    seed = torch_utils.get_numpy(seed)
    pseudo_label = torch_utils.get_numpy(pseudo_label)

    seg_edge = np.max(seed, axis=0)
    seg_edge = (seg_edge * 255).astype(np.uint8)
    seg_edge = cv_utils.get_edge(seg_edge, use_pillow=False)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    seg_edge = cv2.morphologyEx(seg_edge, cv2.MORPH_CLOSE, kernel)
    
    # To avoid segmentation fault, pip install opencv-python==4.5.4.60
    refined_label, ccl_label = cv_utils.get_CCL(255 - seg_edge, pseudo_label)

    return torch.from_numpy(refined_label).long().to(device), \
        [
            torch.from_numpy(seg_edge).long(), 
            torch.from_numpy(ccl_label).long()
        ]

def get_gaussian_blur(_inputs, ksize=5, sigma=0, device=None):
    inputs = _inputs.clone()
    _, c, h, w = inputs.shape

    kernel_1D = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    kernel = kernel_1D * kernel_1D.T

    ksize, ksize = kernel.shape

    kernel = kernel.reshape((1, 1, ksize, ksize))
    kernel = kernel.repeat(c, axis=0)
    kernel = torch.from_numpy(kernel).float().to(device)

    blurred_inputs = F.conv2d(inputs, kernel, padding=ksize//2, groups=c)
    return blurred_inputs

def get_edge_using_laplacian(_inputs, alpha=1, device=None):
    inputs = _inputs.clone()
    _, c, h, w = inputs.shape

    edge_kernel = np.array(
        [
            [-1, -1, -1], 
            [-1,  8, -1], 
            [-1, -1, -1]
        ], 
        dtype=np.float32
    )
    ksize, ksize = edge_kernel.shape

    edge_kernel = edge_kernel.reshape((1, 1, ksize, ksize))
    edge_kernel = edge_kernel.repeat(c, axis=0)
    edge_kernel = torch.from_numpy(edge_kernel).float().to(device)

    edge_mask = F.pad(inputs, pad=tuple([ksize//2]*4), mode='reflect') # edge_mask = inputs
    edge_mask = F.conv2d(edge_mask, edge_kernel, padding=0, groups=c)
    edge_mask = torch.abs(edge_mask)
    
    edge_mask = torch.exp(-alpha * edge_mask)
    edge_mask = 1. - edge_mask
    
    return edge_mask

def get_edge_using_sobel(_inputs, alpha=1, device=None):
    inputs = _inputs.clone()
    _, c, h, w = inputs.shape

    # define kernels
    x_kernel = np.array(
        [
            [+1,  0, -1], 
            [+2,  0, -2], 
            [+1,  0, -1]
        ], 
        dtype=np.float32
    )
    y_kernel = np.array(
        [
            [+1, +2, +1], 
            [ 0,  0,  0], 
            [-1, -2, -1]
        ], 
        dtype=np.float32
    )
    ksize, ksize = x_kernel.shape

    x_kernel = x_kernel.reshape((1, 1, ksize, ksize))
    x_kernel = x_kernel.repeat(c, axis=0)
    x_kernel = torch.from_numpy(x_kernel).float().to(device)

    y_kernel = y_kernel.reshape((1, 1, ksize, ksize))
    y_kernel = y_kernel.repeat(c, axis=0)
    y_kernel = torch.from_numpy(y_kernel).float().to(device)

    edge_x_mask = F.pad(inputs, pad=tuple([ksize//2]*4), mode='reflect') # edge_x_mask = inputs
    edge_x_mask = F.conv2d(edge_x_mask, x_kernel, padding=0, groups=c)
    edge_x_mask = torch.pow(edge_x_mask, 2)

    edge_y_mask = F.pad(inputs, pad=tuple([ksize//2]*4), mode='reflect') # edge_y_mask = inputs
    edge_y_mask = F.conv2d(edge_y_mask, y_kernel, padding=0, groups=c)
    edge_y_mask = torch.pow(edge_y_mask, 2)

    edge_mask = torch.sqrt(edge_x_mask + edge_y_mask)
    edge_mask = torch.exp(-alpha * edge_mask)

    edge_mask = 1. - edge_mask
    
    return edge_mask

def get_edge(_inputs, mode='laplacian', alpha=1, ksize=5, sigma=0, device=torch.device('cuda')):
    if len(_inputs.shape) == 3:
        inputs = _inputs.unsqueeze(0)
    else:
        inputs = _inputs

    if mode == 'laplacian':
        edge = get_edge_using_laplacian(inputs, alpha, device)
    elif mode == 'sobel':
        edge = get_edge_using_sobel(inputs, alpha, device)
    elif mode == 'LoG':
        inputs = get_gaussian_blur(inputs, ksize=ksize, sigma=sigma, device=device)
        edge = get_edge_using_laplacian(inputs, alpha, device)
    elif mode == 'DoG':
        first_inputs = get_gaussian_blur(inputs, ksize=ksize, sigma=sigma*1.6, device=device)
        second_inputs = get_gaussian_blur(inputs, ksize=ksize, sigma=sigma, device=device)

        edge = torch.abs(first_inputs - second_inputs)
        edge = torch.exp(-alpha * edge)
        edge = 1. - edge
    # elif mode == 'SoG':
    #     inputs = get_gaussian_blur(inputs, ksize=ksize, sigma=sigma, device=device)
    #     edge = get_edge_using_sobel(inputs, alpha, device)

    if len(_inputs.shape) == 3:
        edge = edge[0]

    return edge
