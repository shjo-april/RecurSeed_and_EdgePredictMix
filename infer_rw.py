# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import sys
import cv2

import tqdm
import torch
import numpy as np

from core import networks, datasets, inferences, refinements, affinity_utils

from tools.ai import torch_utils, evaluators, augment_utils
from tools.general import io_utils, json_utils, time_utils, cv_utils, pickle_utils

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda', 0)

    # set directories
    model_dir = f'./experiments/models/{args.aff_tag}/'

    if args.strategy == 'last':
        addition_name = ''
    else:
        addition_name = f'@{args.strategy}'
    
    domain = 'train' if args.domain == 'train_aug' else args.domain
    pred_dir = io_utils.create_directory(args.pred_dir + f'predictions/{args.seg_tag}/{domain}/')
    rw_dir = io_utils.create_directory(args.pred_dir + f'random_walk/{args.seg_tag}{addition_name}/{domain}/')

    log_func = lambda string='': print(string)

    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    # create model
    load_fn = lambda strategy: torch_utils.load_model(model, model_dir + f'{strategy}.pth')

    path_index = affinity_utils.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    model = networks.AffinityNet(
        args.backbone, path_index, 
        backbone_norm_fn=args.backbone_norm_fn, 
        aff_norm_fn=args.aff_norm_fn,
        with_instance=args.with_instance
    ).to(device)

    model.eval()

    load_fn(args.strategy)

    log_func('[i] Backbone is {}'.format(args.backbone))
    log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
    log_func()

    # for datasets
    preprocess = [augment_utils.Conditional_Resize(args.image_size)] if args.resize else []
    test_transform = augment_utils.Compose(
        preprocess + [
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

    length = len(test_dataset)
    
    indices = list(range(length))
    if args.reverse:
        indices = indices[::-1]

    if args.shuffle:
        indices = np.asarray(indices[::-1], dtype=np.int32)
        np.random.shuffle(indices)
    
    for i in tqdm.tqdm(indices, desc='Random Walk'):
        image_id, image, label, gt_mask = test_dataset.__getitem__(i)

        if os.path.isfile(rw_dir + f'{image_id}.pkl'):
            continue

        # get keys and seeds
        infer_dict = pickle_utils.load_pickle(pred_dir + image_id + '.pkl')

        class_keys = infer_dict['keys']
        seed = torch.from_numpy(infer_dict['seed']).cuda()

        # random walk
        _, h, w = image.shape
        image = torch.from_numpy(image).cuda()
        
        if len(class_keys) > 1:
            with torch.no_grad():
                aff = model.get_edge(image)

                # print()
                # print(seed.shape, aff.shape)

                rw = torch_utils.resize(seed, aff.size()[-2:])
                rw = affinity_utils.propagate_to_edge(rw, aff, beta=args.beta, exp_times=args.exp_times, radius=args.radius)

                rw = rw[:, 0]
                rw = torch_utils.resize(rw, (h, w))

                rw = rw / torch.max(rw)

                # print(seed.shape, rw.shape)
                # print()

            infer_dict['seed'] = torch_utils.get_numpy(rw)
            
        pickle_utils.dump_pickle(
            rw_dir + f'{image_id}.pkl', 
            infer_dict
        )

if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # environment
    parser.add('gpus', '0', str)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)

    parser.add('pred_dir', './experiments/', str)

    parser.add('domain', 'train_aug', str)
    parser.add('single', False, bool)
    
    # networks
    parser.add('backbone', 'resnet50', str)
    
    parser.add('backbone_norm_fn', 'fix', str)
    parser.add('aff_norm_fn', 'bn', str)

    parser.add('resize', True, bool)
    parser.add('with_instance', False, bool)
    parser.add('image_size', 512, int)

    # RW
    parser.add('beta', 12, int)
    parser.add('exp_times', 8, int)
    parser.add('radius', 5, int)
    
    parser.add('norm', 'max', str)

    # evaluation configuration
    parser.add('aff_tag', 'ResNet50@VOC@AffinityNet', str)
    parser.add('seg_tag', 'ResNet50@VOC@RS+EPM@Official', str)

    parser.add('strategy', 'last', str)

    parser.add('reverse', False, bool)
    parser.add('shuffle', False, bool)

    parser.add('debug', False, bool)
    
    main(parser.get_args())
