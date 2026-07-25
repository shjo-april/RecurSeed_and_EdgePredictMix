# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import torch
import shutil

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from core import networks, datasets

from tools.ai import torch_utils, trainers, optimizers, evaluators, augment_utils
from tools.general import io_utils, log_utils, json_utils

def collate(batch):
    images = []
    masks = []

    for image, mask in batch:
        images.append(torch.from_numpy(image))
        masks.append(torch.from_numpy(mask))
        
    data = {
        'images': torch.stack(images),
        'masks': torch.stack(masks),
    }
    return data

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    gpus = args.gpus.split(',')
    device = torch.device('cuda', 0)

    if RANK in [-1, 0]:
        for i, d in enumerate(gpus):
            p = torch.cuda.get_device_properties(i)
            print(f'[i] CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)')

    if LOCAL_RANK != -1:
        from torch import distributed as dist

        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)

        from datetime import timedelta
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=60))

        assert args.batch_size % WORLD_SIZE == 0, '--batch_size must be multiple of CUDA device count'
    
    # set directories
    txt_dir = './experiments/txt/'
    txt_path = txt_dir + f'{args.tag}.txt'

    model_dir = f'./experiments/models/{args.tag}/'
    tensorboard_dir = f'./experiments/tensorboards/{args.tag}/'

    if os.path.isfile(txt_path) and RANK in [-1, 0]:
        result = input('Found existing logs. yes=remove, no=keyboardinterept')
        if result == 'no':
            raise KeyboardInterrupt
        else:
            if os.path.isfile(txt_path):
                os.remove(txt_path)
            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir)
            if os.path.isdir(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)

    txt_dir = io_utils.create_directory(txt_dir) if RANK in [-1, 0] else txt_dir
    model_dir = io_utils.create_directory(model_dir) if RANK in [-1, 0] else model_dir
    tensorboard_dir = io_utils.create_directory(tensorboard_dir) if RANK in [-1, 0] else tensorboard_dir

    log_func = lambda string='': log_utils.log_print(string, txt_path)

    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    # create model
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
    
    if RANK in [-1, 0]:
        log_func('[i] Backbone is {}'.format(args.backbone))
        log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
        log_func()

    # define loss functions
    loss_fn = nn.CrossEntropyLoss(ignore_index=data_dict['ignore_index']).to(device)

    # define trainer 
    class Trainer(trainers.BaseTrainer):
        def __init__(self):
            param = trainers.Parameter(
                args.seed, 
                True, args.ema_decay,
                args.max_epochs, tensorboard_dir, RANK
            )
            super().__init__(model, device, param)

        def prepare_dataset(self):
            train_transform, valid_transform = \
                augment_utils.get_transform(args)

            if RANK in [-1, 0]:
                log_func('[i] train transform : {}'.format(train_transform))
                log_func('[i] test transform : {}'.format(valid_transform))

            self.train_dataset = datasets.Dataset_For_Segmentation(
                args.root_dir, 
                args.train_domain, 
                train_transform, 
                tag=args.pseudo_tag,
                name=args.dataset,
                mask_dir=args.mask_dir,
            )
            self.valid_dataset = datasets.Dataset_For_Segmentation(
                args.root_dir, 
                args.valid_domain, 
                valid_transform, 
                name=args.dataset
            )
        
        def prepare_loader(self):
            if RANK != -1:
                shuffle = False
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            else:
                shuffle = True
                train_sampler = None
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size // WORLD_SIZE, num_workers=args.num_workers, shuffle=shuffle, drop_last=True, pin_memory=True, sampler=train_sampler, collate_fn=collate)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, num_workers=max(args.num_workers // 4, 1), shuffle=False, drop_last=False, pin_memory=True)

            if RANK in [-1, 0]:
                log_func('[i] The size of training set is {}'.format(len(self.train_dataset)))
                log_func('[i] The size of validation set is {}'.format(len(self.valid_dataset)))

        def configure_optimizers(self):
            self.optimizer = optimizers.SGD(
                params=[
                    {'params': self.param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[1], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[3], 'lr': 10*args.lr, 'weight_decay': args.wd},
                ],
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov,
                scheduler_option={
                    'scheduler': args.scheduler,
                    'power': 0.9,
                    'max_iterations': self.train_iterations * self.param.max_epochs
                }
            )

        def forward(self, data, training: bool=True):
            if training:
                images = data['images'].to(self.device)
                masks = data['masks'].to(self.device)

                output_dict = self.model(images, same=True)
                loss = loss_fn(output_dict['seg_logits'], masks)

                return loss, {
                    'Training/CE': loss.item(),
                    'Training/Learning rate': self.get_learning_rate()
                }
            else:
                images, masks = data

                if self.ema is not None:
                    model = self.ema.get_model()
                else:
                    model = torch_utils.de_parallel(self.model)

                images = images.to(self.device)
                masks = masks.to(self.device)

                _, c, h, w = images.size()

                params = {
                    "image": images[0],
                    "image_size": (h, w),
                    "scales": [1.0],
                    "hflip": False
                }
                output_dict = model.forward_with_scales(**params)

                return {
                    'pred_masks': output_dict['pred_mask'],
                    'gt_masks': masks,
                }

        def evaluation_step(self, debug=False):
            if self.ema is None:
                self.model.eval()
            
            self.evaluator = evaluators.Evaluator_For_Semantic_Segmentation(data_dict['class_names'])
            eval_time = super().evaluation_step(debug)
            valid_mIoU, FP, FN = self.evaluator.get()

            if self.ema is None:
                self.model.train()

            if self.best_valid_score < valid_mIoU:
                self.best_valid_score = valid_mIoU
                self.save_model(model_dir + 'best.pth')

            self.update_tensorboard(
                {
                    'Evaluation/mIoU': valid_mIoU,
                    'Evaluation/Best mIoU': self.best_valid_score,
                    'Evaluation/FP': FP,
                    'Evaluation/FN': FN,
                }
            )

            return {
                'valid_mIoU': valid_mIoU, 
                'best_valid_mIoU': self.best_valid_score, 
                'FP': FP, 
                'FN': FN, 
                'time': eval_time
            }
    
    # train
    trainer = Trainer()

    for epoch in range(trainer.epoch, args.max_epochs+1):
        train_dict = trainer.training_step(args.debug)
        train_dict['epoch'] = epoch
        train_dict['CE'] = train_dict['Training/CE']
        train_dict['lr'] = train_dict['Training/Learning rate']
        
        log_func('[i] epoch={epoch:,}, lr={lr:.6f}, CE={CE:.4f}, time={time:.0f}sec'.format(**train_dict))
        trainer.save_model(model_dir + 'last.pth')

        if epoch % args.valid_ratio == 0:
            valid_dict = trainer.evaluation_step(args.debug)
            valid_dict['epoch'] = epoch

            log_func('[i] epoch={epoch:,}, valid_mIoU={valid_mIoU:.2f}%, FP={FP:.4f}, FN={FN:.4f}, best_valid_mIoU={best_valid_mIoU:.2f}%, time={time:.0f}sec'.format(**valid_dict))

if __name__ == '__main__':
    parser = io_utils.Parser()

    # environment
    parser.add('local_rank', -1, int)

    parser.add('gpus', '0', str)

    parser.add('seed', 1, int)
    parser.add('num_workers', 8, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('mask_dir', './experiments/', str)

    parser.add('train_domain', 'train', str)
    parser.add('valid_domain', 'validation', str)

    parser.add('pseudo_tag', '', str)

    # network architecture
    parser.add('backbone', 'resnet101', str)
    parser.add('decoder', 'deeplabv3+', str)

    parser.add('output_stride', 8, int)
    parser.add('image_size', 512, int)

    # training configuration
    parser.add('batch_size', 16, int)

    parser.add('tag', 'ResNet50@DeepLabv3+@MS@RSCM', str)

    parser.add('max_epochs', 100, int)
    parser.add('valid_ratio', 10.0, float)

    parser.add('lr', 1e-3, float)
    parser.add('wd', 4e-5, float)
    parser.add('momentum', 0.9, float)
    parser.add('nesterov', True, bool)

    parser.add('optimizer', 'SGD', str)
    parser.add('scheduler', 'PolyLR', str)

    parser.add('ema_decay', 0.999, float)

    parser.add('debug', False, bool)

    # data augmentation
    parser.add('min_scale', 0.5, float) 
    parser.add('max_scale', 2.0, float) 

    parser.add('min_image_size', 256, int)
    parser.add('max_image_size', 1024, int)

    parser.add('b_factor', 0.5, float)
    parser.add('c_factor', 0.5, float)
    parser.add('s_factor', 0.5, float)
    parser.add('h_factor', 0.3, float)

    parser.add('zoom_factor', 1, int) # for click annotation

    parser.add('train_augment', 'minmax_scale-hflip-colorjitter-gray-blur-normalize-random_crop-transpose', str) 
    parser.add('test_augment', 'normalize-transpose', str)

    main(parser.get_args())