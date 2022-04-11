# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import sys
import torch

from torch.utils.tensorboard import SummaryWriter

from tools.ai import optimizers
from tools.general import io_utils, time_utils

class Trainer:
    def __init__(self, 
            model, losses, log_names,

            loader, max_epochs,  
            
            optimizer, optimizer_option,
            scheduler, scheduler_option,
            
            amp, ema, tensorboard_dir,

            device, RANK, WORLD_SIZE
        ):

        self.model = model
        self.loader = loader
        self.log_names = log_names

        self.max_epochs = max_epochs
        
        self.num_iterations = len(self.loader)
        self.max_iterations = self.num_iterations * self.max_epochs
        
        if scheduler == 'StepLR':
            scheduler_option['milestones'] = [milestone * self.num_iterations for milestone in scheduler_option['milestones']]
        
        if scheduler in ['StepLR', 'CosineLR']:
            scheduler_option['warmup_iterations'] *= self.num_iterations
        
        scheduler_option['max_iterations'] = self.max_iterations
        optimizer_option['scheduler_option'] = scheduler_option
        
        if optimizer == 'SGD':
            self.optimizer = optimizers.SGD(**optimizer_option)
        elif optimizer == 'AdamW':
            del optimizer_option['momentum']
            del optimizer_option['nesterov']
            self.optimizer = optimizers.AdamW(**optimizer_option)

        self.amp = amp
        self.ema = ema

        self.losses = losses

        self.epoch = 1
        self.iteration = 1

        self.train_timer = time_utils.Timer()
        self.train_meter = io_utils.Average_Meter(self.log_names)
        
        if RANK in [-1, 0]:
            self.writer = SummaryWriter(tensorboard_dir)
        
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        self.device = device

        # for ddp
        self.RANK = RANK
        self.WORLD_SIZE = WORLD_SIZE
    
    def calculate_losses(self, images, labels):
        raise NotImplementedError
    
    def update_tensorboard(self, losses, param_dict):
        raise NotImplementedError

    def update_ema(self):
        if self.ema is not None:
            # print('# update ema')
            self.ema.update(self.model)

    def preprocess(self, data):
        data['images'] = data['images'].to(self.device)
        data['labels'] = data['labels'].to(self.device)
        return data

    def initialize_generator(self):
        self.generator = self.loader

    def save(self, path):
        torch.save(
            {
                'model': self.model.state_dict(),
                'ema': self.ema.get_model().state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path
        )

    def load(self, path):
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict['model'])
        self.ema.get_model().load_state_dict(state_dict['ema'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def step(self, debug=False):
        self.train_timer.tik()
        
        ep_digits = io_utils.get_digits_in_number(self.max_epochs)
        ni_digits = io_utils.get_digits_in_number(self.num_iterations)

        progress_format = '\r# Epoch = %0{}d, [%0{}d/%0{}d] = %02.2f%%, Lr = %.6f, Loss = %.4f\t\t'.format(ep_digits, ni_digits, ni_digits)

        self.initialize_generator()
        
        for i, data in enumerate(self.loader):
            i += 1
            
            # preprocess
            data = self.preprocess(data)
            
            # infer 
            losses, param_dict = self.calculate_losses(data)
            
            # multiply weights for averaging DDP
            # if self.RANK != -1:
            #     losses[0] *= self.WORLD_SIZE
            
            # update weights 
            self.optimizer.zero_grad()
            
            if self.amp:
                self.scaler.scale(losses[0]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses[0].backward()
                self.optimizer.step()
                
            self.update_ema()
            
            # update meter
            self.train_meter.add({name:loss.item() for name, loss in zip(self.log_names, losses)})
            
            # show log
            if self.RANK in [-1, 0]:
                sys.stdout.write(progress_format%(self.epoch, i, self.num_iterations, i / self.num_iterations * 100, self.get_learning_rate(), losses[0].item()))
                sys.stdout.flush()

                # update tensorboard
                if self.iteration % 10 == 0:
                    self.update_tensorboard(losses, param_dict)
            
            self.iteration += 1
            if debug:
                break
        
        print('\r', end='')
        self.epoch += 1
        
        return self.train_meter.get(clear=True), self.train_timer.tok(clear=True)

    def get_writer(self):
        return self.writer
    
    def get_learning_rate(self, option='first'):
        lr_list = [group['lr'] for group in self.optimizer.param_groups]

        if option == 'max':
            return max(lr_list)
        elif option == 'min':
            return min(lr_list)
        elif option == 'first':
            return lr_list[0]
        else:
            return lr_list