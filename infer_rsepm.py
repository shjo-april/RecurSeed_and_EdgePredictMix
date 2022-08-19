# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import sys
import cv2

import tqdm
import torch
import numpy as np

from core import networks, datasets

from tools.ai import torch_utils, augment_utils
from tools.general import io_utils, json_utils, cv_utils, pickle_utils

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda', 0)

    # set directories
    model_dir = f'./experiments/models/{args.tag}/'

    domain = 'train' if args.domain == 'train_aug' else args.domain
    pred_dir = io_utils.create_directory(args.pred_dir + f'predictions/{args.tag}/{domain}/')

    log_func = lambda string='': print(string)

    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    # create model
    load_fn = lambda strategy: torch_utils.load_model(model, model_dir + f'{strategy}.pth')

    model = networks.RSEPM(
        args.backbone, data_dict['num_classes']-1, class_fn=args.class_fn,
        output_stride=16, feature_size=256
    ).to(device)

    model.eval()

    load_fn(args.strategy)

    log_func('[i] Backbone is {}'.format(args.backbone))
    log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
    log_func()

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
    
    scales = [1.0, 0.5, 1.5, 2.0]

    length = len(test_dataset)
    
    indices = list(range(length))

    if args.reverse:
        indices = indices[::-1]

    if args.shuffle:
        indices = np.asarray(indices[::-1], dtype=np.int32)
        np.random.shuffle(indices)
    
    for i in tqdm.tqdm(indices, desc='Inference'):
        image_id, image, label, gt_mask = test_dataset.__getitem__(i)

        if os.path.isfile(pred_dir + f'{image_id}.pkl'):
            continue
    
        # for image_id, image, label, gt_mask in tqdm.tqdm(test_dataset):

        # preprocess
        _, h, w = image.shape
        gt_mask = np.asarray(gt_mask)
        
        image = torch.from_numpy(image).cuda()
        class_mask = torch.from_numpy(label).cuda()
        
        # inference
        params = {
            "image": image,
            "image_size": (h, w),

            "scales": scales,
            "hflip": True,

            "feature_names": ['C1', 'C2', 'C3', 'C4', 'C5'],
            "with_decoder": True,

            "same": args.same,
            "interpolation": args.interpolation
        }
        output_dict = model.forward_with_scales(**params)
        
        if args.conf_th != -1:
            class_mask = (output_dict['pred_class'] >= args.conf_th).float()
        
        output_dict['pred_mask'] = torch_utils.resize(output_dict['pred_mask'], (h, w), mode='bilinear')

        seed = output_dict['pred_mask'][1:]
        seed = torch_utils.normalize(seed)

        class_mask = torch_utils.get_numpy(class_mask)
        seed = torch_utils.get_numpy(seed[class_mask == 1])

        class_keys = np.nonzero(class_mask)[0]
        class_keys = np.pad(class_keys + 1, (1, 0), 'constant')
        
        pickle_utils.dump_pickle(
            pred_dir + f'{image_id}.pkl', 
            {
                'keys': class_keys,
                'seed': seed,
            }
        )

if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # environment
    parser.add('gpus', '0', str)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    
    parser.add('pred_dir', './experiments/', str)

    parser.add('domain', 'train', str)
    parser.add('single', False, bool)
    
    # networks
    parser.add('backbone', 'resnet50', str)
    parser.add('class_fn', 'sigmoid', str)
    parser.add('pool_type', 'gap', str)
    
    parser.add('same', True, bool)
    parser.add('interpolation', 'nearest', str)

    parser.add('conf_th', -1, float)
    parser.add('use_cam', True, bool)
    parser.add('use_scg', True, bool)

    # for SCG
    parser.add('first_th', 0.50, float)
    parser.add('second_th', 0.10, float)
    
    parser.add('foreground_th', 0.30, float)
    parser.add('background_th', 0.05, float)
    
    # evaluation configuration
    parser.add('tag', 'ResNet50@RS(ep=5, cam=1.0, seg=1.0)+CF(c=0.55, u=0.10)+CM(ep=15, cls=1.0, cam=1.0, seg=1.0)', str)
    parser.add('strategy', 'last', str)
    
    parser.add('end_th', 0.70, float)

    parser.add('reverse', False, bool)
    parser.add('shuffle', False, bool)
    
    parser.add('debug', False, bool)
    
    main(parser.get_args())
