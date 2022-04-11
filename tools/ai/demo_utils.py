# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import cv2
import time
import subprocess

import numpy as np

from PIL import Image

from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb

from skimage.segmentation import mark_boundaries

from tools.ai import torch_utils
from tools.general import plot_utils

def add_style(tags, elements):
    if tags is None:
        tags = [None for _ in elements]

    for tag, element in zip(tags, elements):
        if tag is not None:
            write_text(element, tag)
        add_outlier(element)

def get_colors(data_dict):
    ignore_index = data_dict['ignore_index']
    colors = [data_dict['color_dict'][name] for name in data_dict['class_names']]

    while len(colors) <= ignore_index:
        colors.append((0, 0, 0))
    colors[ignore_index] = (224, 224, 192)
    
    colors = np.asarray(colors, dtype=np.uint8)[..., ::-1]
    return colors

def get_bbox_from_mask(pseudo_label, ignore_index=255):
    h, w = pseudo_label.shape
    
    # init bbox
    xmin = 0
    ymin = 0
    xmax = w - 1
    ymax = h - 1
    
    # update bbox
    mask = pseudo_label == ignore_index
    
    if np.sum(mask) > 0:
        # adjust ymin
        while np.sum(mask[ymin, :]) == 0: ymin += 1
        
        # adjust ymax
        while np.sum(mask[ymax, :]) == 0: ymax -= 1

        # adjust xmin
        while np.sum(mask[ymin:ymax, xmin]) == 0: xmin += 1

        # adjust xmax
        while np.sum(mask[ymin:ymax, xmax]) == 0: xmax -= 1

    return xmin, ymin, xmax, ymax

def add_blank(images, length):
    blank = np.zeros_like(images[0])
    for _ in range(length - len(images)):
        images.append(blank)
    return images

def get_gpu_memory_map(device=None):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    
    if device is None:
        return gpu_memory_map
    else:
        return gpu_memory_map[device]

def convert_OpenCV_to_PIL(image):
    return Image.fromarray(image[..., ::-1])

def convert_PIL_to_OpenCV(image):
    return np.asarray(image)[..., ::-1].copy()

def convert_cam(cam):
    cam = (cam * 255).astype(np.uint8)
    if len(cam.shape) == 3:
        cam = np.max(cam, axis=0)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return cam

def add_outlier(image, color=(255, 255, 255)):
    h, w, _ = image.shape
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), color, 2)

def write_text(image, text, fontscale=1.0, bg_color=(0, 255, 0), st_y=0):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fontscale, 1)
    
    cv2.rectangle(image, (0, st_y), (text_width + 5, st_y + text_height + 5), bg_color, cv2.FILLED)
    cv2.putText(image, text, (0, st_y + int(25 * fontscale)), cv2.FONT_HERSHEY_DUPLEX, fontscale, (0, 0, 0), 1)
