# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import tqdm
import torch
import shutil

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from core import networks, datasets, generators, losses, inferences, affinity_utils

from tools.ai import torch_utils, trainers, optimizers, augment_utils
from tools.general import io_utils, log_utils, json_utils

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
    
    args.pretrained_model_path = f'./experiments/models/{args.pseudo_tag}/last.pth'

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
    path_index = affinity_utils.PathIndex(radius=10, default_size=(args.image_size // 4, args.image_size // 4))
    
    model = networks.AffinityNet(
        args.backbone, path_index, 
        backbone_norm_fn=args.backbone_norm_fn, 
        aff_norm_fn=args.aff_norm_fn, with_instance=args.with_instance
    ).to(device)
    
    if RANK in [-1, 0]:
        log_func('[i] Backbone is {}'.format(args.backbone))
        log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
        log_func()

    # load pretrained model
    if args.pretrained_model_path != '':
        torch_utils.load_model(model, args.pretrained_model_path, strict=False)
        if RANK in [-1, 0]:
            log_func('[i] Transfer Learning ({})'.format(args.pretrained_model_path))

    # define trainer 
    class Trainer(trainers.BaseTrainer):
        def __init__(self):
            param = trainers.Parameter(
                args.seed, 
                False, args.ema_decay,
                args.max_epochs, tensorboard_dir, RANK
            )
            super().__init__(model, device, param)

        def prepare_dataset(self):
            train_transform, valid_transform = \
                augment_utils.get_transform(args)

            if RANK in [-1, 0]:
                log_func('[i] train transform : {}'.format(train_transform))
                log_func('[i] test transform : {}'.format(valid_transform))
            
            self.train_dataset = datasets.Dataset_For_RW(
                args.root_dir, 
                args.train_domain, 
                path_index,
                args.pseudo_tag,
                train_transform, 
                name=args.dataset,
                mask_dir=args.mask_dir
            )
            self.valid_dataset = datasets.Dataset_For_Analysis(
                args.root_dir, 
                args.train_domain, 
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
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size // WORLD_SIZE, num_workers=args.num_workers, shuffle=shuffle, drop_last=True, pin_memory=True, sampler=train_sampler)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, num_workers=max(args.num_workers // 4, 1), shuffle=False, drop_last=False, pin_memory=True)

            if RANK in [-1, 0]:
                log_func('[i] The size of training set is {}'.format(len(self.train_dataset)))
                log_func('[i] The size of validation set is {}'.format(len(self.valid_dataset)))

        def configure_optimizers(self):
            if args.scheduler == 'PolyLR':
                scheduler_option = {
                    'scheduler': args.scheduler,
                    'power': 0.9,
                    'max_iterations': self.train_iterations * self.param.max_epochs
                }
            elif args.scheduler == 'StepLR':
                scheduler_option = {
                    'scheduler': args.scheduler,
                    'milestones': [int(milestone) * self.train_iterations for milestone in args.milestones.split(',')],
                    'max_iterations': self.train_iterations * self.param.max_epochs,
                }

            if args.backbone_norm_fn == 'fix':
                params = [
                    {'params': self.param_groups[2], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[3], 'lr': 2*args.lr, 'weight_decay': 0},
                ]
            else:
                params = [
                    {'params': self.param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
                    {'params': self.param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
                ]

            if args.optimizer == 'SGD':
                self.optimizer = optimizers.SGD(
                    params=params,
                    lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov,
                    scheduler_option=scheduler_option
                )
            elif args.optimizer == 'AdamW':
                self.optimizer = optimizers.AdamW(
                    params=params,
                    lr=args.lr, weight_decay=args.wd,
                    scheduler_option=scheduler_option
                )

        def resize_as(self, logits, labels):
            _, TH, TW = labels.size()
            _, _, PH, PW = logits.size()

            if TH != PH or TW != PW:
                labels = labels.unsqueeze(1).float()
                labels = torch_utils.resize(labels, (PH, PW), mode='nearest')
                labels = labels[:, 0].long()

            return labels

        def forward(self, data, training: bool=True):
            if training:
                images, labels = data

                images = images.to(self.device)
                bg_pos_label = labels[0].to(self.device, non_blocking=True)
                fg_pos_label = labels[1].to(self.device, non_blocking=True)
                neg_label = labels[2].to(self.device, non_blocking=True)

                outputs = self.model(images, training=True)

                if len(outputs) == 2: aff, disp = outputs
                else: aff = outputs
                
                pos_aff_loss = (-1) * torch.log(aff + 1e-5)
                neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

                bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
                fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)

                pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
                neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

                loss = (pos_aff_loss + neg_aff_loss) / 2

                if len(outputs) == 2:
                    dp_fg_loss = torch.abs(disp - model.disp_target)
                    dp_bg_loss = torch.abs(disp)

                    dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
                    dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

                    loss += ((dp_fg_loss + dp_bg_loss) / 2)
                else:
                    dp_fg_loss = torch.zeros(1).float().to(self.device)
                    dp_bg_loss = torch.zeros(1).float().to(self.device)
                
                return loss, {
                    'Training/Loss': loss.item(),

                    'Training/Background_Loss': bg_pos_aff_loss.item(),
                    'Training/Foreground_Loss': fg_pos_aff_loss.item(),
                    'Training/Negative_Loss': neg_aff_loss.item(),

                    'Training/Background_DP_Loss': dp_fg_loss.item(),
                    'Training/Foreground_DP_Loss': dp_bg_loss.item(),

                    'Training/Learning rate': self.get_learning_rate()
                }
            else:
                pass
    
    # train
    trainer = Trainer()

    for epoch in range(trainer.epoch, args.max_epochs+1):
        train_dict = trainer.training_step(args.debug)
        train_dict['epoch'] = epoch
        train_dict['Loss'] = train_dict['Training/Loss']

        train_dict['BG_Loss'] = train_dict['Training/Background_Loss']
        train_dict['FG_Loss'] = train_dict['Training/Foreground_Loss']
        train_dict['NEG_Loss'] = train_dict['Training/Negative_Loss']

        train_dict['BG_DP_Loss'] = train_dict['Training/Background_DP_Loss']
        train_dict['FG_DP_Loss'] = train_dict['Training/Foreground_DP_Loss']
        
        train_dict['lr'] = train_dict['Training/Learning rate']
        
        log_func('[i] epoch={epoch:,}, lr={lr:.6f}, Loss={Loss:.4f}, BG_Loss={BG_Loss:.4f}, FG_Loss={FG_Loss:.4f}, NEG_Loss={NEG_Loss:.4f}, BG_DP_Loss={BG_DP_Loss:.4f}, FG_DP_Loss={FG_DP_Loss:.4f}, time={time:.0f}sec'.format(**train_dict))
        
        trainer.save_model(model_dir + 'last.pth')

        if str(epoch) in args.history.split(','):
            trainer.save_model(model_dir + f'ep={epoch}.pth')

    # turn dp mean value from training dataset.
    dp_mean_list = []

    for images, _ in tqdm.tqdm(trainer.valid_loader, desc='Analyzing displacements mean value'):
        images = images.to(device)

        _, dp = model(images, training=False)

        dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())

    trainer.model.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
    trainer.save_model(model_dir + 'last.pth')

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

    parser.add('train_domain', 'train_aug', str)
    parser.add('valid_domain', 'train', str)

    parser.add('single', False, bool)
    parser.add('pseudo_tag', 'ResNet50@VOC@RS+EPM', str)
    parser.add('pretrained_model_path', '', str)
    
    # network architecture
    parser.add('backbone', 'resnet50', str)

    parser.add('backbone_norm_fn', 'fix', str)
    parser.add('aff_norm_fn', 'bn', str)
    parser.add('with_instance', False, bool)

    parser.add('image_size', 512, int)
    
    # training configuration
    parser.add('batch_size', 16, int)

    parser.add('tag', 'ResNet50@VOC@AffinityNet', str)

    parser.add('max_epochs', 30, int)
    parser.add('valid_ratio', 5.0, float)

    parser.add('lr', 1e-1, float)
    parser.add('wd', 4e-5, float)
    parser.add('momentum', 0.0, float)
    parser.add('nesterov', False, bool)

    parser.add('optimizer', 'SGD', str)
    parser.add('scheduler', 'StepLR', str)
    parser.add('milestones', '10,20', str)

    parser.add('ema_decay', 0.999, float)

    parser.add('history', '', str)
    parser.add('debug', False, bool)

    # data augmentation
    parser.add('min_scale', 0.5, float) 
    parser.add('max_scale', 2.0, float) 

    parser.add('min_image_size', 320, int)
    parser.add('max_image_size', 640, int)

    parser.add('b_factor', 0.5, float)
    parser.add('c_factor', 0.5, float)
    parser.add('s_factor', 0.5, float)
    parser.add('h_factor', 0.3, float)

    parser.add('zoom_factor', 1, int) # for click annotation

    parser.add('train_augment', 'minmax_scale-hflip-colorjitter-gray-blur-normalize-random_crop-transpose-maskresize', str) 
    parser.add('test_augment', 'normalize-transpose', str)
    
    main(parser.get_args())