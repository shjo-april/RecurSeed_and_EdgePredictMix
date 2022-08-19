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

from core import datasets, refinements

from tools.ai import torch_utils, augment_utils
from tools.general import io_utils, json_utils, cv_utils, pickle_utils

def main(args):
    if 'MS' in args.tag:
        args.bg_gamma = None

    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda', 0)

    # set directories
    model_dir = f'./experiments/models/{args.tag}/'

    # if args.folder == 'random_walk': extra_info = '@RW'
    # else: extra_info = ''
    
    # if args.bg_gamma is not None:
    #     if args.bg_gamma >= 1:
    #         extra_info += '@G={:d}'.format(int(args.bg_gamma))
    #     else:
    #         extra_info += '@G={:.2f}'.format(args.bg_gamma)
    # elif args.bg_th is not None:
    #     extra_info += '@T={:.2f}'.format(args.bg_th)
    # else:
    #     extra_info = ''

    extra_info = ''
    
    domain = 'train' if args.domain == 'train_aug' else args.domain
    pred_dir = io_utils.create_directory(args.pred_dir + f'{args.folder}/{args.tag}/{domain}/')
    pseudo_dir = io_utils.create_directory(args.pred_dir + f'pseudo-labels/{args.tag}'+ extra_info + f'/{domain}/')

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
            # print('[{}] already'.format(image_id))
            return

        if not os.path.isfile(pred_dir + image_id + '.pkl'):
            print('[{}] not found'.format(image_id))
            return
        
        # preprocess
        _, h, w = image.shape
        gt_mask = np.asarray(gt_mask)
        
        # inference
        try:
            infer_dict = pickle_utils.load_pickle(pred_dir + f'{image_id}.pkl')
        except:
            print('[{}] crash'.format(image_id))
            return

        class_keys = infer_dict['keys']

        if len(class_keys) > 1:
            image = denorm_fn(image).copy()

            ih, iw = image.shape[:2]
            sh, sw = infer_dict['seed'].shape[:2]
            
            if ih != sh or iw != sw:
                seed = torch.from_numpy(infer_dict['seed'])
                seed = torch_utils.resize(seed, (ih, iw))
                infer_dict['seed'] = seed.numpy()
            
            if args.mode == 'crf':
                pseudo_label = inference_for_crf(image, infer_dict['seed'], args.bg_th, args.bg_gamma)
            else:
                h, w, c = image.shape
                
                bg = np.ones((1, h, w), dtype=np.float32) * args.bg_th
                pseudo_label = np.concatenate([bg, infer_dict['seed']], axis=0)

            pseudo_label = np.argmax(pseudo_label, axis=0)
            pseudo_label = class_keys[pseudo_label]
        else:
            pseudo_label = np.zeros((h, w), dtype=np.uint8)

        image = Image.fromarray(pseudo_label.astype(np.uint8)).convert('P')
        image.putpalette(colors[..., ::-1])
        image.save(pseudo_dir + image_id + '.png')
        
        # print('[{}] apply crf'.format(image_id))

    indices = list(range(len(test_dataset)))
    if args.reverse:
        indices = indices[::-1]

    joblib.Parallel(n_jobs=args.num_workers, verbose=10, pre_dispatch="all")(
        [joblib.delayed(postprocess)(i) for i in indices]
    )
    
    # for i in indices: postprocess(i)
    
if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # environment
    parser.add('gpus', '0', str)
    parser.add('num_workers', 16, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('pred_dir', './experiments/', str)

    parser.add('domain', 'train', str)
    parser.add('single', False, bool)

    parser.add('folder', 'predictions', str)
    parser.add('mode', 'crf', str)
    
    # evaluation configuration
    parser.add('tag', 'ResNet50@RS(ep=5, cam=1.0, seg=1.0)+CF(c=0.55, u=0.10)+CM(ep=15, cls=1.0, cam=1.0, seg=1.0)', str)

    parser.add('bg_th', None, float)
    parser.add('bg_gamma', 2.0, float)

    parser.add('reverse', False, bool)
    
    main(parser.get_args())
