# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import sys
import cv2

import tqdm
import torch
import joblib

import numpy as np

from PIL import Image

from core import networks, datasets, inferences, refinements

from tools.ai import torch_utils, evaluators, augment_utils
from tools.general import io_utils, json_utils, time_utils, cv_utils, pickle_utils

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # set directories
    extra_info = '@C={:.2f}@U={:.2f}'.format(args.certain, args.uncertain)

    domain = 'train' if args.domain == 'train_aug' else args.domain
    pred_dir = io_utils.create_directory(args.pred_dir + f'predictions/{args.tag}/{domain}/')
    # pseudo_dir = io_utils.create_directory(args.pred_dir + f'affinity-labels/{args.tag}'+ extra_info + f'/{domain}/')
    pseudo_dir = io_utils.create_directory(args.pred_dir + f'affinity-labels/{args.tag}/{domain}/')

    log_func = lambda string='': print(string)

    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    # for datasets
    test_transform = augment_utils.Compose(
        [
            augment_utils.Normalize(),
            augment_utils.Transpose(),
        ]
    )

    test_dataset = datasets.Dataset_For_Analysis(
        args.root_dir, 
        args.domain, 
        test_transform,
        name=args.dataset,
        single=args.single
    )

    # for evaluation
    colors = cv_utils.get_colors(data_dict)
    denorm_fn = augment_utils.Denormalize()

    crf_dict = {
        'deeplab':{
            'iter_max':10,
            'bi_w':4, 'bi_xy_std':67, 'bi_rgb_std':3, 
            'pos_w':3, 'pos_xy_std':1
        }
    }
    inference_for_crf = refinements.DenseCRF(**crf_dict['deeplab'])

    def postprocess(i):
        image_id, image, label, gt_mask = test_dataset.__getitem__(i)

        pseudo_path = pseudo_dir + image_id + '.png'
        if os.path.isfile(pseudo_path):
            print('[{}] already'.format(image_id))
            return

        if not os.path.isfile(pred_dir + image_id + '.pkl'):
            print('[{}] not found'.format(image_id))
            return
        
        # preprocess
        _, h, w = image.shape
        gt_mask = np.asarray(gt_mask)
        
        # inference
        infer_dict = pickle_utils.load_pickle(pred_dir + f'{image_id}.pkl')

        class_keys = infer_dict['keys']

        if len(class_keys) > 1:
            image = denorm_fn(image).copy()
            
            certain_label = inference_for_crf(image, infer_dict['seed'], None, args.certain)
            certain_label = np.argmax(certain_label, axis=0)
            certain_label = class_keys[certain_label]

            uncertain_label = inference_for_crf(image, infer_dict['seed'], None, args.uncertain)
            uncertain_label = np.argmax(uncertain_label, axis=0)
            uncertain_label = class_keys[uncertain_label]

            affinity_label = certain_label.copy()
            affinity_label[certain_label == 0] = data_dict['ignore_index']
            affinity_label[uncertain_label + certain_label == 0] = 0
        else:
            affinity_label = np.zeros((h, w), dtype=np.uint8)

        image = Image.fromarray(affinity_label.astype(np.uint8)).convert('P')
        image.putpalette(colors[..., ::-1])
        image.save(pseudo_dir + image_id + '.png')
        
        print('[{}] apply crf'.format(image_id))

    indices = list(range(len(test_dataset)))

    joblib.Parallel(n_jobs=args.num_workers, verbose=10, pre_dispatch="all")(
        [joblib.delayed(postprocess)(i) for i in indices]
    )
    
if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # environment
    parser.add('gpus', '0', str)
    parser.add('num_workers', 32, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('pred_dir', './experiments/', str)

    parser.add('domain', 'train', str)
    parser.add('single', False, bool)
    
    # evaluation configuration
    parser.add('tag', 'ResNet50@VOC@RS+EPM', str)
    
    parser.add('certain', 2, float)
    parser.add('uncertain', 32, float)

    main(parser.get_args())
