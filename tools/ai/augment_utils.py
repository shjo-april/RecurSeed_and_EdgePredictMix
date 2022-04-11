# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import cv2
import random
import numpy as np

from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance

from torchvision import transforms

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, output_dict):
        for transform in self.transforms:
            output_dict = transform(output_dict)
        return output_dict
    
    def __repr__(self):
        text = 'Compose(\n'
        for transform in self.transforms:
            text += '\t{}\n'.format(transform)
        text += ')'
        return text

class Conditional_Resize:
    def __init__(self, max_image_size=512):
        self.max_image_size = max_image_size
        
        self.key_dict = {
            'image': Image.BICUBIC, 
            'mask': Image.NEAREST
        }
    
    def resize(self, image, size, mode):
        w, h = image.size

        condition = w < h

        if condition:
            scale = size / h
        else:
            scale = size / w
        
        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h:
            return image
        else:
            return image.resize(size, mode)
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        w, h = image.size

        if max(w, h) > self.max_image_size:
            size = self.max_image_size

            image = self.resize(image, size, self.key_dict['image'])
            if mask is not None:
                mask = self.resize(mask, size, self.key_dict['mask'])
        
        output_dict['image'], output_dict['mask'] = image, mask
        return output_dict

    def __repr__(self):
        return 'Conditional Resize (image_size={})'.format(self.max_image_size)

class Resize:
    def __init__(self, max_image_size=512):
        self.max_image_size = max_image_size
        
        self.key_dict = {
            'image': Image.BICUBIC, 
            'mask': Image.NEAREST
        }
    
    def resize(self, image, size, mode):
        w, h = image.size

        condition = w < h

        if condition:
            scale = size / h
        else:
            scale = size / w
        
        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h:
            return image
        else:
            return image.resize(size, mode)
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        w, h = image.size

        if max(w, h) > self.max_image_size:
            size = self.max_image_size

            image = self.resize(image, size, self.key_dict['image'])
            if mask is not None:
                mask = self.resize(mask, size, self.key_dict['mask'])
        
        output_dict['image'], output_dict['mask'] = image, mask
        return output_dict

    def __repr__(self):
        return 'Resize (image_size={})'.format(self.max_image_size)

class Random_Resize:
    def __init__(self, min_image_size, max_image_size, full=False):
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.full = full
        
        self.key_dict = {
            'image': Image.BICUBIC, 
            'mask': Image.NEAREST
        }

    def get_image_size(self):
        return random.randint(self.min_image_size, self.max_image_size)
    
    def resize(self, image, size, mode):
        w, h = image.size

        if not self.full:
            condition = w < h
        else:
            condition = w > h
        
        if condition:
            scale = size / h
        else:
            scale = size / w
        
        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h:
            return image
        else:
            return image.resize(size, mode)
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        size = self.get_image_size()

        image = self.resize(image, size, self.key_dict['image'])
        if mask is not None:
            mask = self.resize(mask, size, self.key_dict['mask'])
        
        output_dict['image'], output_dict['mask'] = image, mask
        return output_dict

    def __repr__(self):
        return 'Random_Resize ({}, {})'.format(self.min_image_size, self.max_image_size)

class Random_Rescale:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.key_dict = {
            'image': Image.BICUBIC, 
            'mask': Image.NEAREST
        }

    def get_scale(self):
        return self.min_scale + random.random() * (self.max_scale - self.min_scale)
    
    def resize(self, image, scale, mode):
        w, h = image.size

        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h:
            return image
        else:
            return image.resize(size, mode)
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        scale = self.get_scale()

        image = self.resize(image, scale, self.key_dict['image'])
        if mask is not None:
            mask = self.resize(mask, scale, self.key_dict['mask'])
        
        output_dict['image'], output_dict['mask'] = image, mask
        return output_dict

    def __repr__(self):
        return 'Random_Rescale ({}, {})'.format(self.min_scale, self.max_scale)

class Random_HFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        if np.random.rand() <= self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        output_dict['image'], output_dict['mask'] = image, mask
        return output_dict

    def __repr__(self):
        return 'Random_HFlip (p={})'.format(self.p)

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        # 1. pillow to numpy
        image = np.asarray(image, dtype=np.float32)

        # 2. normalize
        norm_image = np.empty_like(image, np.float32)

        norm_image[..., 0] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
        norm_image[..., 1] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
        norm_image[..., 2] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

        if mask is not None:
            mask = np.asarray(mask, dtype=np.int64)

        output_dict['image'], output_dict['mask'] = norm_image, mask
        return output_dict

    def __repr__(self):
        return 'Normalize (mean={}, std={})'.format(self.mean, self.std)

class Denormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # 1. transpose
        # (c, h, w) -> (h, w, c)
        image = image.transpose((1, 2, 0))

        # 2. denormalize
        image = (image * self.std) + self.mean
        image = (image * 255).astype(np.uint8)

        return image

    def __repr__(self):
        return 'Denormaliza (mean={}, std={})'.format(self.mean, self.std)

class Transpose:
    def __init__(self):
        pass
    
    def __call__(self, output_dict):
        output_dict['image'] = output_dict['image'].transpose((2, 0, 1))
        return output_dict
    
    def __repr__(self):
        return 'Transpose (HWC -> CHW)'

class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, output_dict):
        output_dict['image'] = super().__call__(output_dict['image'])
        return output_dict

class Random_Crop:
    def __init__(self, size, channels=3, bg_image=0, bg_mask=255):
        self.bg_image = bg_image
        self.bg_mask = bg_mask

        self.size = size
        self.shape = (size, size, channels)

    def get_random_crop_box(self, image):
        h, w, _ = image.shape
        
        crop_h = min(self.size, h)
        crop_w = min(self.size, w)

        w_space = w - self.size
        h_space = h - self.size

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space + 1)
        else:
            cont_left = random.randrange(-w_space + 1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space + 1)
        else:
            cont_top = random.randrange(-h_space + 1)
            img_top = 0

        dst_bbox = {
            'xmin' : cont_left, 'ymin' : cont_top,
            'xmax' : cont_left+crop_w, 'ymax' : cont_top+crop_h
        }
        src_bbox = {
            'xmin' : img_left, 'ymin' : img_top,
            'xmax' : img_left+crop_w, 'ymax' : img_top+crop_h
        }

        return src_bbox, dst_bbox
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        src_bbox, dst_bbox = self.get_random_crop_box(image)
        
        cropped_image = np.ones(self.shape, image.dtype) * self.bg_image
        cropped_image[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
            image[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
        
        if mask is not None:
            cropped_mask = np.ones(self.shape[:2], image.dtype) * self.bg_mask
            cropped_mask[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
                mask[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
        else:
            cropped_mask = None

        output_dict['image'] = cropped_image
        output_dict['mask'] = cropped_mask
        output_dict['crop_bbox'] = [dst_bbox['xmin'], dst_bbox['ymin'], dst_bbox['xmax'], dst_bbox['ymax']]

        return output_dict

    def __repr__(self):
        return 'RandomCrop (size={})'.format(self.size)

class Resize_For_Mask:
    def __init__(self, size):
        self.size = (size, size)
    
    def __call__(self, output_dict):
        mask = output_dict['mask']

        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize(self.size, Image.NEAREST)
        mask = np.asarray(mask, dtype=np.uint64)

        output_dict['mask'] = mask
        return output_dict
    
    def __repr__(self):
        return 'Resize For Mask ({})'.format(self.size)

##################################################################
# Photometric Transformations
##################################################################
class AutoContrast:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            image = ImageOps.autocontrast(image)
        return image, mask

    def __repr__(self):
        return 'AutoContrast (p={})'.format(self.p)

class Equalize:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            image = ImageOps.equalize(image)
        return image, mask

    def __repr__(self):
        return 'Equalize (p={})'.format(self.p)

class Invert:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            image = ImageOps.invert(image)
        return image, mask

    def __repr__(self):
        return 'Invert (p={})'.format(self.p)

class Posterize:
    def __init__(self, p=0.5, max_v=4, fix=False):
        self.p = p

        self.min_v = max_v if fix else 1
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            v = np.random.randint(self.min_v, self.max_v + 1)
            v = max(v, 2)
            image = ImageOps.posterize(image, v)
        return image, mask

    def __repr__(self):
        return 'Posterize (p={})'.format(self.p)

class Solarize:
    def __init__(self, p=0.5, max_v=256, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            v = np.random.randint(self.min_v, self.max_v + 1)
            image = ImageOps.solarize(image, v)
        return image, mask

    def __repr__(self):
        return 'Solarize (p={})'.format(self.p)

class SolarizeAdd:
    def __init__(self, p=0.5, v=128, max_addition=110, fix=False):
        self.p = p

        self.v = v

        self.min_addition = max_addition if fix else 0
        self.max_addition = max_addition
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            addition = np.random.randint(self.min_addition, self.max_addition + 1)

            np_image = np.array(image).astype(np.int32)

            np_image = np_image + addition
            np_image = np.clip(np_image, 0, 255)
            np_image = np_image.astype(np.uint8)

            image = Image.fromarray(np_image)
            image = ImageOps.solarize(image, self.v)
        
        return image, mask

    def __repr__(self):
        return 'SolarizeAdd (p={})'.format(self.p)

class Color:
    def __init__(self, p=0.5, max_v=1.9, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0.1
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            v = np.random.uniform(self.min_v, self.max_v)
            image = ImageEnhance.Color(image).enhance(v)
        return image, mask

    def __repr__(self):
        return 'Color (p={})'.format(self.p)

class Contrast:
    def __init__(self, p=0.5, max_v=1.9, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0.1
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            v = np.random.uniform(self.min_v, self.max_v)
            image = ImageEnhance.Contrast(image).enhance(v)
        return image, mask

    def __repr__(self):
        return 'Contrast (p={})'.format(self.p)

class Brightness:
    def __init__(self, p=0.5, max_v=1.9, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0.1
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            v = np.random.uniform(self.min_v, self.max_v)
            image = ImageEnhance.Brightness(image).enhance(v)
        return image, mask

    def __repr__(self):
        return 'Brightness (p={})'.format(self.p)

class Sharpness:
    def __init__(self, p=0.5, max_v=1.9, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0.1
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:
            v = np.random.uniform(self.min_v, self.max_v)
            image = ImageEnhance.Sharpness(image).enhance(v)
        return image, mask

    def __repr__(self):
        return 'Sharpness (p={})'.format(self.p)

class Photometric_Transform:
    def __init__(self, num_augments=2, magnitude=9):
        self.n = num_augments
        self.m = magnitude

        self.colorjitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)

        self.ops = [
            AutoContrast(p=1.0),
            Equalize(p=1.0),
            Invert(p=1.0),

            Posterize(p=1.0, max_v=(self.m / 30.) * 4., fix=True),
            Solarize(p=1.0, max_v=(self.m / 30.) * 256, fix=True),
            SolarizeAdd(p=1.0, max_addition=(self.m / 30.) * 110, fix=True),

            Color(p=1.0, max_v=(self.m / 30.) * 1.8 + 0.1, fix=True),
            Contrast(p=1.0, max_v=(self.m / 30.) * 1.8 + 0.1, fix=True),
            Brightness(p=1.0, max_v=(self.m / 30.) * 1.8 + 0.1, fix=True),
            Sharpness(p=1.0, max_v=(self.m / 30.) * 1.8 + 0.1, fix=True),
        ]

    def __call__(self, image):
        image, _ = self.colorjitter(image)

        for op in random.choices(self.ops):
            image, _ = op(image)

        return image

##################################################################
# Geometric Transformations
##################################################################
class Rotate:
    def __init__(self, p=0.5, max_angle=30, fix=False):
        self.p = p

        self.min_angle = max_angle if fix else 0
        self.max_angle = max_angle
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:

            v = np.random.randint(self.min_angle, self.max_angle + 1)
            if np.random.rand() <= 0.5:
                v = -v
            
            image = image.rotate(v)
        return image, mask

    def __repr__(self):
        return 'Rotate (p={})'.format(self.p)

class ShearX:
    def __init__(self, p=0.5, max_v=0.3, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:

            v = np.random.uniform(self.min_v, self.max_v)
            if np.random.rand() <= 0.5:
                v = -v
            
            image = image.transform(image.size, Image.AFFINE, (1, v, 0, 0, 1, 0))
        return image, mask

    def __repr__(self):
        return 'ShearX (p={})'.format(self.p)

class ShearY:
    def __init__(self, p=0.5, max_v=0.3, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:

            v = np.random.uniform(self.min_v, self.max_v)
            if np.random.rand() <= 0.5:
                v = -v
            
            image = image.transform(image.size, Image.AFFINE, (1, 0, 0, v, 1, 0))
        return image, mask

    def __repr__(self):
        return 'ShearY (p={})'.format(self.p)

class TranslateXabs:
    def __init__(self, p=0.5, max_v=100, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:

            v = np.random.randint(self.min_v, self.max_v + 1)
            if np.random.rand() <= 0.5:
                v = -v
            
            image = image.transform(image.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
        return image, mask

    def __repr__(self):
        return 'TranslateXabs (p={})'.format(self.p)

class TranslateYabs:
    def __init__(self, p=0.5, max_v=100, fix=False):
        self.p = p

        self.min_v = max_v if fix else 0
        self.max_v = max_v
    
    def __call__(self, image, mask=None):
        if np.random.rand() <= self.p:

            v = np.random.randint(self.min_v, self.max_v + 1)
            if np.random.rand() <= 0.5:
                v = -v
            
            image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
        return image, mask

    def __repr__(self):
        return 'TranslateYabs (p={})'.format(self.p)
