# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import cv2
import cmapy
import numpy as np

from typing import Tuple
from PIL import ImageFont, ImageDraw, Image

ESC = 27
SPACEBAR = 32

def create_color_maps(N = 256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    
    return cmap

def thresholding(mask, low_th=0.10, sigma=0.25):
    v = np.median(mask[mask >= int(255 * low_th)])
    upper = int(min(255, (1.0 + sigma) * v))
    
    condition = mask >= upper
    # condition = mask >= low_th

    # cv2.imshow('condition', (condition*255).astype(np.uint8))
    # cv2.waitKey(0)

    mask[condition] = 255
    mask[np.logical_not(condition)] = 0

    return mask

def get_kmeans(image, K, attempts=10):
    image = np.asarray(image)
    h, w, c = image.shape

    flat_image = np.reshape(image, (h*w, c)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(flat_image, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # center = center.astype(np.uint8)
    # result = center[label.flatten()]
    # result = result.reshape(image.shape)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # cv2.imshow('Image, Result', np.concatenate([image, result], axis=1))
    # cv2.waitKey(0)

    label = np.reshape(label, (h, w))
    label = label.astype(np.uint8)

    print(label.dtype, label.min(), label.max(), label.shape)

    return label

def get_contours(binary_mask: np.asarray):
    contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea)[::-1]

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area <= 0.0:
            continue

        result = np.zeros_like(binary_mask)
        cv2.drawContours(result, [contour], 0, (255,255,255), cv2.FILLED)

        cv2.imshow('Result', result)
        cv2.waitKey(0)

def get_CCL(binary_mask: np.asarray, pseudo_label: np.asarray, ignore_th: float=0.9):
    num_labels, mask, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)

    # collect candidate labels
    labels = []
    areas = []

    for label, stat in enumerate(stats):
        area = stat[4]
        # if area < (5*5):
        #     continue

        labels.append(label)
        areas.append(area)

    labels = np.asarray(labels)
    areas = np.asarray(areas)

    labels = labels[np.argsort(areas)[::-1]]

    # visualize
    ccl_index = 1
    ccl_label = np.zeros_like(pseudo_label)

    refined_label = pseudo_label.copy()
    
    find_edge = False

    for label in labels:
        ccl_mask = mask == label

        if not find_edge:
            edge = np.sum(ccl_mask * (binary_mask == 255))
            if edge == 0:
                edge_mask = np.logical_and(ccl_mask, pseudo_label == 0)
                refined_label[edge_mask] = 255

                find_edge = True
                continue
        
        values, counts = np.unique(pseudo_label[ccl_mask], return_counts=True)
        ratios = counts / np.sum(counts)
        # print(values, counts)

        bg_ratio = 0
        for v, r in zip(values, ratios):
            if v in [0, 255]:
                bg_ratio += r

        ccl_label[ccl_mask] = ccl_index
        ccl_index += 1

        if len(values) > 1:
            max_index = np.argmax(counts)
            value = values[max_index]

            """ v1
            if value == 0 and bg_ratio < ignore_th:
                continue
            
            if not value in [0, 255]:
                # can_mask = np.logical_or(pseudo_label == 0, pseudo_label == 255)
                can_mask = pseudo_label == 255
                refined_label[np.logical_and(can_mask, ccl_mask)] = value
            else:
                can_mask = np.logical_or(
                    pseudo_label == 255, 
                    pseudo_label == 0,
                )
                refined_label[np.logical_and(can_mask, ccl_mask)] = value

                can_mask = np.logical_and(
                    pseudo_label != 255, 
                    pseudo_label != 0,
                )
                refined_label[np.logical_and(can_mask, ccl_mask)] = 255
            """

            # v2 - simple
            can_mask = pseudo_label == 255
            refined_label[np.logical_and(can_mask, ccl_mask)] = value

            # refined_label[ccl_mask] = value

            # cv2.imshow('CCL', (ccl_mask*255).astype(np.uint8))
            # cv2.waitKey(0)
    
    return refined_label, ccl_label

def get_edge(image, mode='canny', use_pillow=True):
    if use_pillow:
        image = np.asarray(image)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # sigma = 0.25
    # v = np.median(image)

    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))

    lower = 10
    upper = 100

    edge = cv2.Canny(image, lower, upper)

    if use_pillow:
        edge = Image.fromarray(edge)

    return edge

def get_colors(data_dict):
    ignore_index = data_dict['ignore_index']
    colors = [data_dict['color_dict'][name] for name in data_dict['class_names']]

    while len(colors) <= ignore_index:
        colors.append((0, 0, 0))
    colors[ignore_index] = (224, 224, 192)
    
    # RGB to BGR
    colors = np.asarray(colors, dtype=np.uint8)[..., ::-1]
    return colors

def apply_colormap(cam, option='SEISMIC'):
    color_dict = {
        'JET': cv2.COLORMAP_JET,
        'HOT': cv2.COLORMAP_HOT,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'WINTER': cv2.COLORMAP_WINTER,
        'SEISMIC': cmapy.cmap('seismic'),
    }

    if cam.dtype in [np.float32, np.float64]:
        cam = (cam * 255).astype(np.uint8)
    
    if len(cam.shape) == 3:
        cam = np.max(cam, axis=0)
    
    if option in color_dict:
        cam = cv2.applyColorMap(cam, color_dict[option])
    return cam

def add_blank(images, length):
    blank = np.zeros_like(images[0])
    for _ in range(length - len(images)):
        images.append(blank)
    return images

def draw_text_using_opencv(image: np.ndarray, text: str, coordinate: Tuple[int, int], color: Tuple[int, int, int]):
    cv2.putText(image, text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)    

def draw_text_using_pillow(image: np.ndarray, text: str, coordinate: Tuple[int, int], color: Tuple[int, int, int]=(0, 0, 0), font_path: str='./data/Pretendard-Regular.otf', font_size: int=20, background: tuple=(79, 244, 255), centering: bool=False):
    text = ' ' + text
    font = ImageFont.truetype(font_path, font_size)
    
    tw, th = font.getsize(text)
    if centering:
        coordinate = list(coordinate)
        coordinate[0] -= tw // 2
        coordinate[1] -= th // 2
        coordinate = tuple(coordinate)

    if background is not None:
        cv2.rectangle(image, coordinate, (coordinate[0] + tw + 5, coordinate[1] + th + 5), background, cv2.FILLED)
    
    pillow_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pillow_image)
    draw.text(coordinate, text, font=font, fill=(color[0], color[1], color[2], 0))

    image[:, :, :] = np.asarray(pillow_image)

def rotation_using_pillow(image: np.ndarray, angle: int):
    pillow_image = Image.fromarray(image)
    pillow_image = pillow_image.rotate(angle, expand=True)
    return np.asarray(pillow_image)