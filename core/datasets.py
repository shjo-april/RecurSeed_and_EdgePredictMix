# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

from .abc_modules import Guide_For_Dataset

from tools.ai import torch_utils
from tools.general import cv_utils
from tools.general import json_utils

class Dataset_For_Classification(Guide_For_Dataset):
    def __init__(self, root_dir, domain, transform=None, single=False, with_crop_bbox=False, name='VOC'):
        super().__init__(root_dir, domain, ['image', 'tag'], f'./data/{name}.json')
        
        self.transform = transform
        self.with_crop_bbox = with_crop_bbox

        if single:
            self.image_names = self.get_single_labels()

    def get_single_labels(self):
        single_image_names = []

        for image_name in self.image_names:
            image_id = image_name.replace('.jpg', '')
            tags = self.get_tag(image_id)
            
            if len(tags) == 1:
                single_image_names.append(image_name)
        
        return single_image_names
    
    def __getitem__(self, index):
        data = super().__getitem__(index)

        image = data['image']

        if self.transform is not None:
            input_dict = {'image': image, 'mask':None}
            output_dict = self.transform(input_dict)

            image = output_dict['image']
        
        label = torch_utils.one_hot_embedding([self.class_dict[tag] - 1 for tag in data['tag']], self.num_classes - 1)

        if self.with_crop_bbox: return image, label, output_dict['crop_bbox']
        else: return image, label

class Dataset_For_Analysis(Guide_For_Dataset):
    def __init__(self, root_dir, domain, transform=None, name='VOC', single=False):
        super().__init__(root_dir, domain, ['image_id', 'image', 'tag', 'mask'], f'./data/{name}.json')

        self.name = name
        self.transform = transform

        if single:
            self.image_names = self.get_single_labels()

    def get_single_labels(self):
        single_image_names = []

        for image_name in self.image_names:
            image_id = image_name.replace('.jpg', '')
            tags = self.get_tag(image_id)
            
            if len(tags) == 1:
                single_image_names.append(image_name)
        
        return single_image_names

    def __getitem__(self, index):
        data = super().__getitem__(index)

        input_dict = {'image': data['image'], 'mask': data['mask']}

        if self.transform is not None:
            output_dict = self.transform(input_dict)
        else:
            output_dict = input_dict

        image_id = data['image_id']
        image, mask = output_dict['image'], output_dict['mask']

        # For VOC and COCO (including background class)
        if self.name in ['VOC', 'COCO']:
            label = torch_utils.one_hot_embedding([self.class_dict[tag] - 1 for tag in data['tag']], self.num_classes - 1)
        
        return image_id, image, label, mask
        
    def get_from_image_id(self, image_id, apply_transform=False):
        data = super().get_data_from_image_id(image_id)

        image_id = data['image_id']
        image = data['image']
        mask = data['mask']

        input_dict = {
            'image': data['image'], 
            'mask': data['mask'],
            # 'edge': cv_utils.get_edge(data['image'], 'canny')
            # 'kmean': cv_utils.get_kmeans(data['image'], 100)
        }

        if self.transform is not None and apply_transform:
            output_dict = self.transform(input_dict)
        else:
            output_dict = input_dict
        
        image, mask = output_dict['image'], output_dict['mask']

        # For VOC and COCO (including background class)
        if self.name in ['VOC', 'COCO']:
            label = torch_utils.one_hot_embedding([self.class_dict[tag] - 1 for tag in data['tag']], self.num_classes - 1)

        return image_id, image, label, mask #, output_dict['edge']

class Dataset_For_Segmentation(Guide_For_Dataset):
    def __init__(self, root_dir, domain, transform, tag=None, name='VOC', mask_dir='./experiments/'):
        super().__init__(root_dir, domain, ['image', 'mask'], f'./data/{name}.json')
        self.transform = transform

        if tag is not None:
            self.mask_dir = mask_dir + 'pseudo-labels/{}/train/'.format(tag)

    def __getitem__(self, index):
        data = super().__getitem__(index)

        input_dict = {'image': data['image'], 'mask': data['mask']}
        output_dict = self.transform(input_dict)
        
        return output_dict['image'], output_dict['mask']

class Dataset_For_Visualization(Guide_For_Dataset):
    def __init__(self, root_dir, domain, transform=None, name='VOC'):
        super().__init__(root_dir, domain, ['image_id', 'image', 'tag', 'mask', 'bbox'], f'./data/{name}.json')

        self.name = name
        self.transform = transform
    
    def __getitem__(self, index):
        pass
        
    def get_from_image_id(self, image_id, apply_transform=False):
        data = super().get_data_from_image_id(image_id)

        image_id = data['image_id']
        image = data['image']
        mask = data['mask']

        bboxes = []
        for (xmin, ymin, xmax, ymax), tag in zip(*data['bbox']):
            bboxes.append([xmin, ymin, xmax, ymax, tag])

        input_dict = {
            'image': data['image'], 
            'mask': data['mask'],
        }

        if self.transform is not None and apply_transform:
            output_dict = self.transform(input_dict)
        else:
            output_dict = input_dict
        
        image, mask = output_dict['image'], output_dict['mask']

        # For VOC and COCO (including background class)
        if self.name in ['VOC', 'COCO']:
            label = torch_utils.one_hot_embedding([self.class_dict[tag] - 1 for tag in data['tag']], self.num_classes - 1)

        return image_id, image, label, mask, bboxes