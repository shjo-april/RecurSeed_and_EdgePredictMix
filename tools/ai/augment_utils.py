# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import cv2
import torch
import random
import numpy as np

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageEnhance

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform(args):
    augment_dict = {
        'resize':Resize(args.image_size),
        'minmax_scale':Random_Rescale(args.min_scale, args.max_scale),
        'minmax_resize':Random_Resize(args.min_image_size, args.max_image_size),
        
        'hflip':Random_HFlip(),
        'blur':Random_GaussianBlur(p=0.5, kernel=(5, 5)),
        'gray':Random_Grayscale(p=0.2),
        'rotation': Random_Rotation(degree=10),
        
        'colorjitter':ColorJitter(brightness=args.b_factor, contrast=args.c_factor, saturation=args.s_factor, hue=args.h_factor),
        
        'normalize':Normalize(IMAGENET_MEAN, IMAGENET_STD),
        'transpose':Transpose(),
        
        'random_crop':Random_Crop(args.image_size),
        'center_crop':Center_Crop(args.image_size),

        'click2mask': Clicks2Mask(ignore_index=255, zoom_factor=args.zoom_factor),
        'maskresize': Resize_For_Mask(args.image_size // 4)
    }

    train_transforms = []
    for name in args.train_augment.split('-'):
        if name in augment_dict.keys():
            transform = augment_dict[name]
        else:
            raise ValueError('unrecognize name of transform ({})'.format(name))
        
        train_transforms.append(transform)

    test_transforms = []
    for name in args.test_augment.split('-'):
        if name in augment_dict.keys():
            transform = augment_dict[name]
        else:
            raise ValueError('unrecognize name of transform ({})'.format(name))
        
        test_transforms.append(transform)

    train_transform = Compose(train_transforms)
    test_transform = Compose(test_transforms)

    return train_transform, test_transform

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

class Fixed_Resize:
    def __init__(self, image_size: tuple):
        self.image_size = image_size
        
        self.key_dict = {
            'image': Image.BICUBIC, 
            'mask': Image.NEAREST
        }
    
    def resize(self, image, mode):
        return image.resize(self.image_size, mode)
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        image = self.resize(image, self.key_dict['image'])
        if mask is not None:
            mask = self.resize(mask, self.key_dict['mask'])
    
        output_dict['image'], output_dict['mask'] = image, mask
        return output_dict

    def __repr__(self):
        return 'Fixed_Resize (image_size={})'.format(self.image_size)

class Conditional_Resize:
    def __init__(self, max_image_size=512, with_mask=True):
        self.with_mask = with_mask
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
            if mask is not None and self.with_mask:
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

class Random_Rotation:
    def __init__(self, degree=10, p=0.5, ignore_index=255, mean=[0.485, 0.456, 0.406]):
        self.p = p
        self.degree = degree
        self.ignore_index = ignore_index

        self.key_dict = {
            'image': Image.BICUBIC, 
            'mask': Image.NEAREST,
            'uncertain_mask': Image.BILINEAR
        }

        self.fill_dict = {
            'image': tuple([int(m * 255) for m in mean]),
            'mask': self.ignore_index,
            'uncertain_mask': (1 * 255)
        }

    def __call__(self, output_dict):
        if np.random.rand() <= self.p:
            deg = np.random.randint(-self.degree, self.degree+1, 1)[0]
            
            for key in output_dict.keys():
                # print(key, output_dict[key])
                if key == 'clicks':
                    continue

                if key in self.fill_dict:
                    output_dict[key] = output_dict[key].rotate(deg, self.key_dict[key], fillcolor=self.fill_dict[key])
                else:
                    output_dict[key] = output_dict[key].rotate(deg, Image.BILINEAR, fillcolor=0)
        
        return output_dict

    def __repr__(self):
        return 'Random_Rotation (degree={})'.format(self.degree)

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

        resized_image = self.resize(image, size, self.key_dict['image'])
        if mask is not None:
            mask = self.resize(mask, size, self.key_dict['mask'])
        
        output_dict['image'], output_dict['mask'] = resized_image, mask
        return output_dict

    def __repr__(self):
        return 'Random_Resize ({}, {})'.format(self.min_image_size, self.max_image_size)

class Random_Rescale:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.key_dict = {
            'image': Image.BICUBIC, 
            'mask': Image.NEAREST,
            'uncertain_mask': Image.BILINEAR,
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
        scale = self.get_scale()

        for key in output_dict.keys():
            if output_dict[key] is not None:
                if key == 'clicks':
                    continue

                if key in self.key_dict:
                    output_dict[key] = self.resize(output_dict[key], scale, self.key_dict[key])
                else:
                    output_dict[key] = self.resize(output_dict[key], scale, Image.BILINEAR)
        
        return output_dict

    def __repr__(self):
        return 'Random_Rescale ({}, {})'.format(self.min_scale, self.max_scale)

class Random_HFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, output_dict):
        if np.random.rand() <= self.p:
            for key in output_dict.keys():
                if key == 'clicks':
                    clicks = output_dict['clicks']
                    for click in clicks:
                        click['x'] = 1. - click['x']    
                    output_dict['clicks'] = clicks

                elif output_dict[key] is not None:
                    output_dict[key] = output_dict[key].transpose(Image.FLIP_LEFT_RIGHT)

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

        for key in output_dict.keys():
            if not key in ['image', 'mask', 'clicks']:
                output_dict[key] = np.asarray(output_dict[key], dtype=np.float32) # / 255

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

    def to(self, image, device=torch.device('cuda')):
        mean = torch.Tensor(self.mean).float().to(device)
        std = torch.Tensor(self.std).float().to(device)

        image = image * std[:, None, None] + mean[:, None, None]
        # print(image.shape, image.min(), image.max())
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

class Random_Grayscale(transforms.RandomGrayscale):
    def __init__(self, p=0.2):
        super().__init__(p)

    def __call__(self, output_dict):
        output_dict['image'] = super().__call__(output_dict['image'])
        return output_dict

class Random_GaussianBlur(transforms.GaussianBlur):
    def __init__(self, p=0.5, kernel=(5, 5)):
        super().__init__(kernel)
        self.p = p

    def __call__(self, output_dict):
        if np.random.rand() <= self.p:
            output_dict['image'] = super().__call__(output_dict['image'])
        return output_dict

class Random_Crop:
    def __init__(self, size, channels=3, bg_image=0, bg_mask=255):
        self.bg_image = bg_image
        self.bg_mask = bg_mask

        if isinstance(size, int):
            self.w_size = size
            self.h_size = size
        elif isinstance(size, tuple):
            self.w_size, self.h_size = size

        self.shape = (self.h_size, self.w_size, channels)

    def get_random_crop_box(self, image):
        h, w, _ = image.shape
        
        crop_h = min(self.h_size, h)
        crop_w = min(self.w_size, w)

        w_space = w - self.w_size
        h_space = h - self.h_size

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
            cropped_mask = np.ones(self.shape[:2], mask.dtype) * self.bg_mask
            cropped_mask[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
                mask[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
        else:
            cropped_mask = None

        if 'uncertain_mask' in output_dict:
            umask = output_dict['uncertain_mask']
            cropped_umask = np.zeros(self.shape[:2], umask.dtype)
            cropped_umask[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
                umask[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
            output_dict['uncertain_mask'] = cropped_umask

        for key in output_dict.keys():
            if not key in ['image', 'mask', 'uncertain_mask', 'clicks']:
                tmask = output_dict[key]

                cropped_tmask = np.zeros(self.shape[:2], umask.dtype)
                cropped_tmask[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
                    tmask[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
                
                output_dict[key] = cropped_tmask

        output_dict['image'] = cropped_image
        output_dict['mask'] = cropped_mask
        output_dict['crop_bbox'] = [dst_bbox['xmin'], dst_bbox['ymin'], dst_bbox['xmax'], dst_bbox['ymax']]

        return output_dict

    def __repr__(self):
        return 'Random_Crop (size={})'.format((self.w_size, self.h_size))

class Center_Crop:
    def __init__(self, size, channels=3, bg_image=0, bg_mask=255):
        self.bg_image = bg_image
        self.bg_mask = bg_mask

        if isinstance(size, int):
            self.w_size = size
            self.h_size = size
        elif isinstance(size, tuple):
            self.w_size, self.h_size = size

        self.shape = (self.h_size, self.w_size, channels)

    def get_center_box(self, image):
        h, w, _ = image.shape
        
        crop_h = min(self.h_size, h)
        crop_w = min(self.w_size, w)

        dst_xmin = self.w_size // 2 - crop_w // 2
        dst_ymin = self.h_size // 2 - crop_h // 2

        src_xmin = w // 2 - crop_w // 2
        src_ymin = h // 2 - crop_h // 2

        # print(src_xmin, src_ymin)
        # print(dst_xmin, dst_ymin)
        
        dst_bbox = {
            'xmin' : dst_xmin, 'ymin' : dst_ymin,
            'xmax' : dst_xmin+crop_w, 'ymax' : dst_ymin+crop_h
        }
        src_bbox = {
            'xmin' : src_xmin, 'ymin' : src_ymin,
            'xmax' : src_xmin+crop_w, 'ymax' : src_ymin+crop_h
        }

        return src_bbox, dst_bbox
    
    def __call__(self, output_dict):
        image, mask = output_dict['image'], output_dict['mask']

        src_bbox, dst_bbox = self.get_center_box(image)
        
        cropped_image = np.ones(self.shape, image.dtype) * self.bg_image
        cropped_image[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
            image[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
        
        if mask is not None:
            cropped_mask = np.ones(self.shape[:2], mask.dtype) * self.bg_mask
            cropped_mask[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = \
                mask[src_bbox['ymin']:src_bbox['ymax'], src_bbox['xmin']:src_bbox['xmax']]
        else:
            cropped_mask = None

        output_dict['image'] = cropped_image
        output_dict['mask'] = cropped_mask
        output_dict['crop_bbox'] = [dst_bbox['xmin'], dst_bbox['ymin'], dst_bbox['xmax'], dst_bbox['ymax']]

        return output_dict

    def __repr__(self):
        return 'CenterCrop (size={})'.format((self.w_size, self.h_size))

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

class Clicks2Mask:
    def __init__(self, ignore_index, zoom_factor=1):
        self.ignore_index = ignore_index
        self.zoom_factor = zoom_factor

    def __call__(self, output_dict):
        w, h = output_dict['image'].size
        rw, rh = w // self.zoom_factor, h // self.zoom_factor

        click_mask = np.ones((rh, rw), dtype=np.uint8) 
        click_mask *= self.ignore_index
        
        for click in output_dict['clicks']:
            y, x = click['y'], click['x']

            y = min(int(y * rh), rh-1)
            x = min(int(x * rw), rw-1)
            
            click_mask[y, x] = click['class_index']

        if self.zoom_factor > 1:
            click_mask = cv2.resize(click_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        click_mask = Image.fromarray(click_mask)
        output_dict['mask'] = click_mask
        return output_dict

    def __repr__(self):
        return 'Click2Mask()'
