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

    if args.decoder == 'deeplabv3+':
        model = networks.DeepLabv3_Plus(
            args.backbone, 
            data_dict['num_classes'],
            output_stride=args.output_stride
        ).to(device)
    elif args.decoder == 'deeplabv2':
        model = networks.DeepLabv2(
            args.backbone, 
            data_dict['num_classes'],
            output_stride=args.output_stride
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
    
    # for image_id, image, label, gt_mask in tqdm.tqdm(test_dataset):
    for i in tqdm.tqdm(indices, desc='Inference'):
        image_id, image, label, gt_mask = test_dataset.__getitem__(i)

        if os.path.isfile(pred_dir + f'{image_id}.pkl'):
            continue

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
            "hflip": True
        }
        output_dict = model.forward_with_scales(**params)
        
        output_dict['pred_mask'] = torch_utils.resize(output_dict['pred_mask'], (h, w), mode='bilinear')

        pred_mask = torch_utils.get_numpy(output_dict['pred_mask'])
        class_mask = np.max(pred_mask.reshape(data_dict['num_classes'], h*w), axis=1) >= 0.05

        # print(class_mask.shape, pred_mask.shape)

        class_keys = np.nonzero(class_mask)[0]
        seed = pred_mask[0, class_mask, :, :]
        
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
    parser.add('backbone', 'resnet101', str)
    parser.add('decoder', 'deeplabv3+', str)

    parser.add('output_stride', 8, int)
    
    # evaluation configuration
    parser.add('tag', 'ResNet101@VOC@DeepLabv3+@MS@RSCM', str)
    parser.add('strategy', 'last', str)

    parser.add('reverse', False, bool)
    parser.add('shuffle', False, bool)

    parser.add('debug', False, bool)
    
    main(parser.get_args())
