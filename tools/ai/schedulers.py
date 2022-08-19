# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import numpy as np

class _Scheduler:
    def __init__(self, optimizer, max_iterations):
        self.optimizer = optimizer

        self.iteration = 1
        self.max_iterations = max_iterations

    def state_dict(self) -> dict:
        return {
            'iteration': self.iteration
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.iteration = state_dict['iteration']

    def get_learning_rate(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def set_learning_rate(self, lrs):
        if isinstance(lrs, float):
            lrs = [lrs for _ in range(len(self.optimizer.param_groups))]
        
        for lr, group in zip(lrs, self.optimizer.param_groups):
            group['lr'] = lr

class PolyLR(_Scheduler):
    def __init__(self, optimizer, max_iterations, power=0.9):
        super().__init__(optimizer, max_iterations)
        
        self.power = power
        self.init_lrs = self.get_learning_rate()
    
    def step(self):
        if self.iteration < self.max_iterations:
            lr_mult = (1 - self.iteration / self.max_iterations) ** self.power

            lrs = [lr * lr_mult for lr in self.init_lrs]
            self.set_learning_rate(lrs)

            self.iteration += 1

class StepLR(_Scheduler):
    def __init__(self, optimizer, max_iterations, milestones, warmup_iterations=0):
        super().__init__(optimizer, max_iterations)
        
        self.milestones = milestones
        self.lrs = self.get_learning_rate()

        self.warmup_lrs = [lr * 0.1 for lr in self.lrs]
        self.warmup_iterations = warmup_iterations

        if self.warmup_iterations > 0:
            self.set_learning_rate(self.warmup_lrs)

    def apply_warmup(self):
        lrs = []
        for lr, warmup_lr in zip(self.lrs, self.warmup_lrs):
            lr = warmup_lr + (lr - warmup_lr) * (self.iteration / self.warmup_iterations)
            lrs.append(lr)
        return lrs

    def step(self):
        if self.warmup_iterations > 0 and self.iteration <= self.warmup_iterations:
            lrs = self.apply_warmup()
        else:
            if self.iteration in self.milestones:
                self.lrs = [lr * 0.1 for lr in self.lrs]
            lrs = self.lrs
        
        self.set_learning_rate(lrs)
        self.iteration += 1

class CosineLR(_Scheduler):
    def __init__(self, optimizer, max_iterations, warmup_iterations=0):
        super().__init__(optimizer, max_iterations)

        self.lrs = self.get_learning_rate()
        self.decay_iterations = max_iterations - warmup_iterations
        
        self.warmup_lrs = [lr * 0.1 for lr in self.lrs]
        self.warmup_iterations = warmup_iterations

        if self.warmup_iterations > 0:
            self.set_learning_rate(self.warmup_lrs)

    def apply_warmup(self):
        lrs = []
        for lr, warmup_lr in zip(self.lrs, self.warmup_lrs):
            lr = warmup_lr + (lr - warmup_lr) * (self.iteration / self.warmup_iterations)
            lrs.append(lr)
        return lrs

    def apply_cosine(self):
        lrs = []
        for lr in self.lrs:
            lr = 0.5 * lr * (1 + np.cos(np.pi * (self.iteration - self.warmup_iterations) / self.decay_iterations))
            lrs.append(lr)
        return lrs

    def step(self):
        if self.warmup_iterations > 0 and self.iteration <= self.warmup_iterations:
            lrs = self.apply_warmup()
        else:
            lrs = self.apply_cosine()
        
        self.set_learning_rate(lrs)
        self.iteration += 1
