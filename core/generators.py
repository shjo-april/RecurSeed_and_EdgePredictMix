# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from .refinements import PAMR

from core import inferences, edge_modules, layers
from tools.ai import torch_utils, augment_utils

class RecurSeed:
    def __init__(self, 
            device, background=255, 
            ema=None, scales=[1.0, 0.5, 1.5], # for cam
            scg_dict={}, # for scg
            refinement='pamr', # for refinement
            gamma=1.0, # for uncertainty
            interpolation='bilinear', same=False
        ):
        self.ema = ema

        self.scales = scales
        self.hflip = True

        self.device = device
        self.background = background

        self.gamma = gamma
        self.interpolation = interpolation

        self.inference_for_cam = inferences.CAM()

        self.inference_for_scg = inferences.SCG(**scg_dict)
        self.inference_for_scg.to(self.device)

        if refinement == 'pamr':
            self.inference = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24], sigma=0.1)
            self.inference.to(device)
        else:
            self.inference = None

        self.same = same
        self.with_decoder = False
        self.denorm_fn = augment_utils.Denormalize()

    def resize(self, image, image_size):
        OH, OW = image_size

        _, CH, CW = image.size()

        if CW < CH:
            scale = OH / CH
        else:
            scale = OW / CW

        return torch_utils.resize(image, (int(round(CH * scale)), int(round(CW * scale))))
    
    def normalize(self, x, eps=1e-5):
        C, H, W = x.size()
        max_x = x.view(C, (H * W)).max(dim=1)[0].view(C, 1, 1)
        return x / (max_x + eps)

    def __call__(self, image, label, crop_bbox, with_cam=False):
        oh, ow = image.size()[1:]
        
        xmin, ymin, xmax, ymax = crop_bbox
        ch, cw = ymax-ymin, xmax-xmin
        
        cropped_image = image[:, ymin:ymax, xmin:xmax]

        with torch.no_grad():
            _image = self.resize(cropped_image, (oh, ow))

            params = {
                "image": _image,
                "image_size": _image.size()[1:],

                "scales": self.scales,
                "hflip": self.hflip,

                "feature_names": ['C1', 'C2', 'C3', 'C4', 'C5'],
                "interpolation": self.interpolation,

                "with_decoder": self.with_decoder,
                "same": self.same
            }
            output_dict = self.ema.get_model().forward_with_scales(**params)

            # for cam
            seed = self.inference_for_cam(output_dict['pred_cams'], 'add')
            seed = seed[label == 1]

            cam = seed.clone()

            # for scg
            seed = self.inference_for_scg(
                seed.unsqueeze(0),
                output_dict['features']
            )[0]

            seed = torch_utils.resize(seed, (ch, cw), mode=self.interpolation)

            # for a seed from decoder
            if self.with_decoder:
                pred_mask = torch_utils.resize(output_dict['pred_mask'], (ch, cw), mode=self.interpolation)    
                seed_from_decoder = self.normalize(pred_mask[1:][label == 1]) # renorm: marginally reduce FN
                seed_from_decoder = pred_mask[1:][label == 1]
            
            if self.inference is not None:
                # print(self.denorm_fn.to(cropped_image).unsqueeze(0).shape)
                # print(seed.unsqueeze(0).shape)

                seed = self.inference(
                    self.denorm_fn.to(cropped_image).unsqueeze(0), 
                    seed.unsqueeze(0)
                )[0]
                seed = torch.clip(seed, min=0.0, max=1.0)

                if self.with_decoder:
                    seed_from_decoder = self.inference(
                        self.denorm_fn.to(cropped_image).unsqueeze(0), 
                        seed_from_decoder.unsqueeze(0)
                    )[0]
                    seed_from_decoder = torch.clip(seed_from_decoder, min=0.0, max=1.0)
        
        ## generate certain mask
        # certain_mask = torch.zeros((oh, ow)).float().to(self.device)
        # certain_mask[ymin:ymax, xmin:xmax] = torch.pow(torch.max(seed, dim=0)[0], self.gamma)

        ## generate uncertain mask
        # uncertain_mask = torch.zeros((oh, ow)).float().to(self.device)        
        # uncertain_mask[ymin:ymax, xmin:xmax] = 1. - certain_mask[ymin:ymax, xmin:xmax] 
        
        seed = torch_utils.normalize(seed) # renorm: marginally reduce FN

        rs_dict = {
            'seeds_from_classifier': seed,
            # 'certain_mask': certain_mask,
            # 'uncertain_mask': uncertain_mask,
        }

        if with_cam:
            rs_dict['cams'] = torch_utils.resize(cam, (ch, cw), mode='nearest')

        if self.with_decoder:
            rs_dict['seeds_from_decoder'] = seed_from_decoder
            rs_dict['pred_mask'] = pred_mask

        return rs_dict

class CertainFilter:
    def __init__(self, 
            device, background,
            certain_th, uncertain_th
        ):
        self.device = device
        self.background = background

        self.certain_th = certain_th
        self.uncertain_th = uncertain_th
        
    def __call__(self, image, seed, crop_bbox, label, use_edge=False):
        class_keys = torch.nonzero(label)[:, 0]
        class_keys = F.pad(class_keys + 1, (1, 0), 'constant')

        OH, OW = image.size()[1:]
        C, CH, CW = seed.size()

        xmin, ymin, xmax, ymax = crop_bbox

        pseudo_label = torch.ones((OH, OW)).long().to(self.device) * self.background
        pseudo_label[ymin:ymax, xmin:xmax] = 0
        
        # 5.1. apply uncertain region to avoid calculating loss
        if self.uncertain_th > 0:
            max_seed = torch.max(seed, dim=0)[0]
            pseudo_label[ymin:ymax, xmin:xmax][max_seed >= self.uncertain_th] = self.background
        
        # 5.2. apply certain region
        certain_label = torch.zeros((class_keys.size(0), CH, CW)).float().to(self.device)
        certain_label[0] = self.certain_th
        certain_label[1:] = seed
        
        certain_label = torch.argmax(certain_label, dim=0)
        certain_label = class_keys[certain_label]
        
        certain_mask = certain_label != 0
        pseudo_label[ymin:ymax, xmin:xmax][certain_mask] = certain_label[certain_mask]

        output_dict = {'pseudo_label': pseudo_label}

        if use_edge:
            refined_label, (partial_seg_edge, partial_ccl_label) = edge_modules.refine_label_using_edge(
                seed, 
                pseudo_label[ymin:ymax, xmin:xmax]
            )
            pseudo_label[ymin:ymax, xmin:xmax] = refined_label

            seg_edge = torch.zeros((OH, OW)).long()
            seg_edge[ymin:ymax, xmin:xmax] = partial_seg_edge

            ccl_label = torch.zeros((OH, OW)).long()
            ccl_label[ymin:ymax, xmin:xmax] = partial_ccl_label

            output_dict['seg_edge'] = seg_edge
            output_dict['ccl_label'] = ccl_label
        
        return output_dict

class FillBackground:
    def __init__(self, device, background):
        self.device = device
        self.background = background

    def __call__(self, image, seed, crop_bbox, label):
        OH, OW = image.size()[1:]
        xmin, ymin, xmax, ymax = crop_bbox

        pseudo_label = torch.ones((len(label), OH, OW)).float().to(self.device) * self.background
        pseudo_label[label == 0, ymin:ymax, xmin:xmax] = 0.
        pseudo_label[label == 1, ymin:ymax, xmin:xmax] = seed 

        return pseudo_label

class EdgePredictMix:
    def __init__(self, device, background, num_classes=20, ce_strategy='opq'):
        self.device = device
        self.background = background

        self.num_classes = num_classes
        self.ce_strategy = ce_strategy
    
    def __call__(self, images, pseudo_labels, seeds):
        B = images.size(0)

        # generate certain masks
        masks = []
        
        for pseudo_label in pseudo_labels:
            mask = (pseudo_label != 0) & (pseudo_label != self.background)
            mask = mask.float().unsqueeze(0)
            masks.append(mask)
        
        masks = torch.stack(masks)

        # shuffle
        indices = np.arange(B)
        np.random.shuffle(indices)

        certain_masks = masks[indices]
        certain_images = images[indices]
        certain_labels = pseudo_labels[indices]
        certain_seeds = seeds[indices]

        # mix
        mixed_images = certain_masks * certain_images + (1. - certain_masks) * images
        mixed_seg_labels = certain_masks[:, 0] * certain_labels + (1. - certain_masks[:, 0]) * pseudo_labels

        mixed_seeds = certain_masks * certain_seeds + (1. - certain_masks) * seeds

        mixed_weights = torch.ones(B, self.num_classes).float().to(self.device)
        mixed_cls_labels = torch.zeros(B, self.num_classes).float().to(self.device)

        np_seg_labels = torch_utils.get_numpy(pseudo_labels)
        np_mixed_seg_labels = torch_utils.get_numpy(mixed_seg_labels)

        for b in range(B):
            for c in range(1, self.num_classes+1):
                seg_sum = (np_seg_labels[b] == c).sum()
                mix_sum = (np_mixed_seg_labels[b] == c).sum()
                
                if seg_sum > 0 or mix_sum > 0:
                    if seg_sum == 0 and mix_sum > 0:
                        mixed_cls_labels[b, c-1] = 1
                    
                    elif seg_sum > 0 and seg_sum > mix_sum:
                        removed_p = (seg_sum - mix_sum) / seg_sum
                        mixed_cls_labels[b, c-1] = 1. - removed_p

                    elif seg_sum > 0 and seg_sum <= mix_sum:
                        mixed_cls_labels[b, c-1] = 1
        
        return {
            'mixed_images': mixed_images,
            'mixed_seeds': mixed_seeds,
            'mixed_cls_labels': mixed_cls_labels,
            'mixed_seg_labels': mixed_seg_labels.long(),
            'mixed_weights': mixed_weights,
        }

class Generator_For_RS:
    def __init__(self, device, ignore_index, rs_params, cf_params):
        self.device = device
        self.ignore_index = ignore_index

        rs_params['device'] = device
        rs_params['background'] = ignore_index
        self.rs = RecurSeed(**rs_params)

        cf_params['device'] = device
        cf_params['background'] = ignore_index
        self.cf = CertainFilter(**cf_params)

        self.bf = FillBackground(device, ignore_index)

        self.with_decoder = False

    def use_decoder(self):
        self.with_decoder = True
        self.rs.with_decoder = True
    
    def __call__(self, images, labels, crop_bboxes, debug=False):
        images = images.to(self.device)
        labels = labels.to(self.device)

        data = {
            'pseudo_labels_from_rs': [],
            'seeds_from_classifier': []
        }
        if self.with_decoder:
            data['seeds_from_decoder'] = []

        for b in range(images.size(0)):
            image = images[b]
            label = labels[b]
            crop_bbox = crop_bboxes[b]
            
            if label.sum().item() > 0:
                rs_dict = self.rs(image, label, crop_bbox)
                cf_dict = self.cf(image, rs_dict['seeds_from_classifier'], crop_bbox, label)

                seeds_from_classifier = self.bf(image, rs_dict['seeds_from_classifier'], crop_bbox, label)
                pseudo_label_from_rs = cf_dict['pseudo_label']

                if self.with_decoder:
                    seeds_from_decoder = self.bf(image, rs_dict['seeds_from_decoder'], crop_bbox, label)
            else:
                _, OH, OW = image.size()
                pseudo_label_from_rs = torch.ones((OH, OW)).long().to(self.device) * self.cf.background
                seeds_from_classifier = torch.ones((len(label), OH, OW)).long().to(self.device) * self.cf.background
                if self.with_decoder:
                    seeds_from_decoder = torch.ones((len(label), OH, OW)).long().to(self.device) * self.cf.background

                xmin, ymin, xmax, ymax = crop_bbox

                pseudo_label_from_rs[ymin:ymax, xmin:xmax] = 0
                seeds_from_classifier[:, ymin:ymax, xmin:xmax] = 0
                if self.with_decoder:
                    seeds_from_decoder[:, ymin:ymax, xmin:xmax] = 0

            data['pseudo_labels_from_rs'].append(pseudo_label_from_rs)
            data['seeds_from_classifier'].append(seeds_from_classifier)

            if self.with_decoder:
                data['seeds_from_decoder'].append(seeds_from_decoder)
        
        data['pseudo_labels_from_rs'] = torch.stack(data['pseudo_labels_from_rs'])
        data['seeds_from_classifier'] = torch.stack(data['seeds_from_classifier'])

        if self.with_decoder:
            data['seeds_from_decoder'] = torch.stack(data['seeds_from_decoder'])
        
        return data

class Generator_For_RSEPM:
    def __init__(self, device, ignore_index, rs_params, cf_params, cm_params):
        self.device = device
        self.ignore_index = ignore_index

        rs_params['device'] = device
        rs_params['background'] = ignore_index
        self.rs = RecurSeed(**rs_params)

        cf_params['device'] = device
        cf_params['background'] = ignore_index
        self.cf = CertainFilter(**cf_params)

        self.bf = FillBackground(device, ignore_index)

        cm_params['device'] = device
        cm_params['background'] = ignore_index

        self.cm = EdgePredictMix(**cm_params)

        self.use_edge = False
        self.with_decoder = False

        self.cm_key = 'seeds_from_classifier' # 'seeds_from_decoder'

    def use_decoder(self):
        self.with_decoder = True
        self.rs.with_decoder = True
    
    def __call__(self, images, labels, crop_bboxes, with_cm=False, debug=False):
        images = images.to(self.device)
        labels = labels.to(self.device)

        data = {
            'pseudo_labels_from_rs': [],
            'seeds_from_classifier': []
        }
        if self.with_decoder:
            data['seeds_from_decoder'] = []
            if with_cm:
                data['pseudo_labels_from_decoder'] = []

        for b in range(images.size(0)):
            image = images[b]
            label = labels[b]
            crop_bbox = crop_bboxes[b]
            
            if label.sum().item() > 0:
                rs_dict = self.rs(image, label, crop_bbox)
                cf_dict = self.cf(image, rs_dict['seeds_from_classifier'], crop_bbox, label)

                seeds_from_classifier = self.bf(image, rs_dict['seeds_from_classifier'], crop_bbox, label)
                pseudo_label_from_rs = cf_dict['pseudo_label']

                if self.with_decoder:
                    seeds_from_decoder = self.bf(image, rs_dict['seeds_from_decoder'], crop_bbox, label)
                    if with_cm:
                        cf_dict = self.cf(image, rs_dict['seeds_from_decoder'], crop_bbox, label, self.use_edge)
                        pseudo_label_from_decoder = cf_dict['pseudo_label']
            else:
                _, OH, OW = image.size()
                pseudo_label_from_rs = torch.ones((OH, OW)).long().to(self.device) * self.cf.background
                if with_cm:
                    pseudo_label_from_decoder = torch.ones((OH, OW)).long().to(self.device) * self.cf.background
                
                seeds_from_classifier = torch.ones((len(label), OH, OW)).long().to(self.device) * self.cf.background
                if self.with_decoder:
                    seeds_from_decoder = torch.ones((len(label), OH, OW)).long().to(self.device) * self.cf.background

                xmin, ymin, xmax, ymax = crop_bbox
                
                pseudo_label_from_rs[ymin:ymax, xmin:xmax] = 0
                if with_cm:
                    pseudo_label_from_decoder[ymin:ymax, xmin:xmax] = 0

                seeds_from_classifier[:, ymin:ymax, xmin:xmax] = 0
                if self.with_decoder:
                    seeds_from_decoder[:, ymin:ymax, xmin:xmax] = 0

            data['pseudo_labels_from_rs'].append(pseudo_label_from_rs)
            data['seeds_from_classifier'].append(seeds_from_classifier)

            if self.with_decoder:
                data['seeds_from_decoder'].append(seeds_from_decoder)
                if with_cm:
                    data['pseudo_labels_from_decoder'].append(pseudo_label_from_decoder)
        
        data['pseudo_labels_from_rs'] = torch.stack(data['pseudo_labels_from_rs'])
        data['seeds_from_classifier'] = torch.stack(data['seeds_from_classifier'])
        
        if self.with_decoder:
            data['seeds_from_decoder'] = torch.stack(data['seeds_from_decoder'])
            if with_cm:
                data['pseudo_labels_from_decoder'] = torch.stack(data['pseudo_labels_from_decoder'])

                cm_dict = self.cm(
                    images, 
                    data['pseudo_labels_from_decoder'], 
                    data[self.cm_key]
                )
                
                data['mixed_images'] = cm_dict['mixed_images']
                data['mixed_seeds'] = cm_dict['mixed_seeds']
                data['mixed_weights'] = cm_dict['mixed_weights']
                data['mixed_cls_labels'] = cm_dict['mixed_cls_labels']
                data['mixed_seg_labels'] = cm_dict['mixed_seg_labels']
        
        return data