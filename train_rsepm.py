# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import torch
import shutil

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from core import networks, datasets, generators, losses, inferences

from tools.ai import torch_utils, trainers, optimizers, augment_utils
from tools.general import io_utils, log_utils, json_utils

def collate(batch):
    images = []
    labels = []
    crop_bboxes = []

    for image, label, crop_bbox in batch:
        images.append(torch.from_numpy(image))
        labels.append(torch.from_numpy(label))
        crop_bboxes.append(crop_bbox)
    
    data = {
        'images': torch.stack(images),
        'labels': torch.stack(labels),
        'crop_bboxes': np.stack(crop_bboxes),
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
    model = networks.RSEPM(
        args.backbone, data_dict['num_classes']-1, class_fn=args.class_fn,
        output_stride=16, feature_size=args.feature_size
    ).to(device)

    # torch_utils.load_model(model, model_dir + 'ep=15.pth')
    
    if RANK in [-1, 0]:
        log_func('[i] Backbone is {}'.format(args.backbone))
        log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
        log_func()

    # define loss functions
    inference_for_cam = inferences.CAM()

    if args.class_fn == 'softmax': class_loss_fn = nn.CrossEntropyLoss(ignore_index=data_dict['ignore_index']).to(device)
    else: class_loss_fn = nn.MultiLabelSoftMarginLoss().to(device)

    cam_loss_fn = losses.L1Loss(
        ignore_index=data_dict['ignore_index'], 
        bg_mode=args.cam_bg_mode,
        balance=args.cam_balance
    ).to(device)
    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=data_dict['ignore_index']).to(device)
    
    # define trainer 
    class Trainer(trainers.BaseTrainer):
        def __init__(self):
            param = trainers.Parameter(
                args.seed, 
                True, args.ema_decay,
                args.max_epochs, tensorboard_dir, RANK
            )
            super().__init__(model, device, param)

            self.generator = generators.Generator_For_RSEPM(
                device=device,
                ignore_index=data_dict['ignore_index'],
                rs_params={ 
                    'ema': self.ema,
                    'scales': list(map(float, args.scales.split(','))),
                    'scg_dict': {
                        'first_th': args.first_th,
                        'second_th': args.second_th,
                        'foreground_th': args.foreground_th,
                        'background_th': args.background_th
                    },
                    'refinement': args.refinement,
                    'interpolation': args.interpolation,
                    'same': args.same
                },
                cf_params={
                    'certain_th': args.certain_th,
                    'uncertain_th': args.uncertain_th,
                },
                cm_params={
                    'num_classes': data_dict['num_classes']-1,
                }
            )
            self.generator.cm_key = args.epm_key

        def prepare_dataset(self):
            train_transform, valid_transform = \
                augment_utils.get_transform(args)

            if RANK in [-1, 0]:
                log_func('[i] train transform : {}'.format(train_transform))
                log_func('[i] test transform : {}'.format(valid_transform))

            self.train_dataset = datasets.Dataset_For_Classification(
                args.root_dir, 
                args.train_domain, 
                train_transform, 
                single=args.single,
                with_crop_bbox=True,
                name=args.dataset,
            )
            self.valid_dataset = datasets.Dataset_For_Analysis(
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

            if args.optimizer == 'SGD':
                self.optimizer = optimizers.SGD(
                    params=[
                        {'params': self.param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
                        {'params': self.param_groups[1], 'lr': args.lr, 'weight_decay': args.wd},
                        {'params': self.param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
                        {'params': self.param_groups[3], 'lr': 10*args.lr, 'weight_decay': args.wd},
                    ],
                    lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov,
                    scheduler_option=scheduler_option
                )
            elif args.optimizer == 'AdamW':
                self.optimizer = optimizers.AdamW(
                    params=[
                        {'params': self.param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
                        {'params': self.param_groups[1], 'lr': args.lr, 'weight_decay': args.wd},
                        {'params': self.param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
                        {'params': self.param_groups[3], 'lr': 10*args.lr, 'weight_decay': args.wd},
                    ],
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
                images = data['images'].to(self.device)
                labels = data['labels'].to(self.device)

                with_rs = self.epoch > args.rs_epoch
                with_epm = self.epoch > args.epm_epoch

                cond_rs_cam = with_rs and args.lambda_rs_cam > 0
                cond_rs_seg = with_rs and args.lambda_rs_seg > 0

                cond_epm_cam = with_epm and args.lambda_epm_cam > 0
                cond_epm_seg = with_epm and args.lambda_epm_seg > 0

                if cond_rs_cam or cond_rs_seg:
                    if with_epm:
                        self.generator.use_edge = args.use_edge
                        self.generator.use_decoder()

                    pseudo_dict = self.generator(images, data['labels'], data['crop_bboxes'], with_epm)
                    for key in pseudo_dict.keys():
                        pseudo_dict[key] = pseudo_dict[key].detach().to(device)

                    if args.debug:
                        print(with_rs)
                        print(with_epm)
                        print(pseudo_dict.keys())
                        print()

                if args.class_fn == 'softmax':
                    labels = torch.argmax(labels, dim=1)

                output_dict = self.model(images, with_decoder=cond_rs_seg)
                if with_epm:
                    mix_output_dict = self.model(pseudo_dict['mixed_images'], with_decoder=cond_epm_seg)

                loss = torch.zeros(1).to(self.device).float()

                class_loss = class_loss_fn(output_dict['cls_logits'], labels)
                loss += class_loss

                if cond_rs_cam:
                    output_dict['cams'] = inference_for_cam(output_dict['cams'], mode='single')

                    rs_cam_loss = args.lambda_rs_cam * cam_loss_fn(
                        output_dict['cams'],
                        pseudo_dict['seeds_from_classifier'],
                        labels
                    )
                    loss += rs_cam_loss
                else:
                    rs_cam_loss = torch.zeros(1).to(self.device).float()

                if cond_rs_seg:
                    rs_seg_loss = args.lambda_rs_seg * seg_loss_fn(
                        output_dict['seg_logits'], 
                        self.resize_as(
                            output_dict['seg_logits'],
                            pseudo_dict['pseudo_labels_from_rs'], 
                        )
                    )
                    loss += rs_seg_loss
                else:
                    rs_seg_loss = torch.zeros(1).to(self.device).float()

                if cond_epm_cam:
                    mix_output_dict['cams'] = inference_for_cam(mix_output_dict['cams'], mode='single')

                    mix_cam_loss = args.lambda_epm_cam * cam_loss_fn(
                        mix_output_dict['cams'],
                        pseudo_dict['mixed_seeds'],
                        pseudo_dict['mixed_cls_labels']
                    )
                    loss += mix_cam_loss
                else:
                    mix_cam_loss = torch.zeros(1).to(self.device).float()

                if cond_epm_seg:
                    mix_seg_loss = args.lambda_epm_seg * seg_loss_fn(
                        mix_output_dict['seg_logits'], 
                        self.resize_as(
                            mix_output_dict['seg_logits'],
                            pseudo_dict['mixed_seg_labels'], 
                        )
                    )
                    loss += mix_seg_loss
                else:
                    mix_seg_loss = torch.zeros(1).to(self.device).float()

                return loss, {
                    'Training/Loss': loss.item(),

                    'Training/CE': class_loss.item(),
                    'Training/RS_CAM_Loss': rs_cam_loss.item(),
                    'Training/RS_SEG_Loss': rs_seg_loss.item(),

                    'Training/EPM_CAM_Loss': mix_cam_loss.item(),
                    'Training/EPM_SEG_Loss': mix_seg_loss.item(),

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

        train_dict['CE'] = train_dict['Training/CE']
        train_dict['RS_CAM_Loss'] = train_dict['Training/RS_CAM_Loss']
        train_dict['RS_SEG_Loss'] = train_dict['Training/RS_SEG_Loss']
        
        train_dict['EPM_CAM_Loss'] = train_dict['Training/EPM_CAM_Loss']
        train_dict['EPM_SEG_Loss'] = train_dict['Training/EPM_SEG_Loss']

        train_dict['lr'] = train_dict['Training/Learning rate']
        
        log_func('[i] epoch={epoch:,}, lr={lr:.6f}, Loss={Loss:.4f}, CE={CE:.4f}, RS_CAM_Loss={RS_CAM_Loss:.4f}, RS_SEG_Loss={RS_SEG_Loss:.4f}, EPM_CAM_Loss={EPM_CAM_Loss:.4f}, EPM_SEG_Loss={EPM_SEG_Loss:.4f},time={time:.0f}sec'.format(**train_dict))
        
        trainer.save_model(model_dir + 'last.pth')

        if str(epoch) in args.history.split(','):
            trainer.save_model(model_dir + f'ep={epoch}.pth')

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

    parser.add('train_domain', 'train', str)
    parser.add('valid_domain', 'validation', str)

    parser.add('single', False, bool)
    
    # network architecture
    parser.add('backbone', 'resnet50', str)
    parser.add('class_fn', 'sigmoid', str)

    parser.add('feature_size', 256, int)
    parser.add('image_size', 512, int)
    
    # training configuration
    parser.add('batch_size', 16, int)

    parser.add('tag', 'ResNet50@VOC@RS+EPM', str)

    parser.add('max_epochs', 30, int)
    parser.add('valid_ratio', 5.0, float)

    parser.add('lr', 1e-1, float)
    parser.add('wd', 4e-5, float)
    parser.add('momentum', 0.0, float)
    parser.add('nesterov', False, bool)

    parser.add('optimizer', 'SGD', str)
    parser.add('scheduler', 'StepLR', str)
    parser.add('milestones', '5', str)

    parser.add('ema_decay', 0.999, float)

    parser.add('history', '5,15', str)
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

    parser.add('train_augment', 'minmax_resize-hflip-normalize-random_crop-transpose', str) 
    parser.add('test_augment', 'normalize-transpose', str)

    # RecurSeed
    parser.add('rs_epoch', 5, int)

    parser.add('lambda_rs_cam', 1, float)
    parser.add('lambda_rs_seg', 1, float)

    parser.add('first_th', 0.50, float)
    parser.add('second_th', 0.10, float)
    parser.add('foreground_th', 0.30, float)
    parser.add('background_th', 0.05, float)

    parser.add('certain_th', 0.55, float)
    parser.add('uncertain_th', 0.10, float)

    parser.add('cam_bg_mode', 'ignore', str)
    parser.add('cam_balance', 'fg+bg', str)

    parser.add('scales', '1.0,0.5,1.5,2.0', str)
    parser.add('refinement', 'pamr', str)

    parser.add('same', False, bool)
    parser.add('interpolation', 'nearest', str)
    
    # EdgePredictMix
    parser.add('epm_epoch', 15, int)
    parser.add('use_edge', True, bool)
    
    parser.add('lambda_epm_cam', 1, float)
    parser.add('lambda_epm_seg', 1, float)
    
    parser.add('epm_key', 'seeds_from_classifier', str)

    main(parser.get_args())