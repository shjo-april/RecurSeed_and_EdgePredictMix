# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

from torch import optim
from .schedulers import PolyLR
from .schedulers import StepLR

class SGD(optim.SGD):
    def __init__(self, params, lr, weight_decay, momentum, nesterov, scheduler_option):
        super().__init__(params, lr=lr, momentum=momentum, nesterov=nesterov)
        
        name = scheduler_option['scheduler']
        del scheduler_option['scheduler']

        self.scheduler = eval(name)(self, **scheduler_option)
    
    def step(self, closure=None):
        super().step(closure)
        self.scheduler.step()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict['schedule'])
        del state_dict['schedule']
        return super().load_state_dict(state_dict)
    
    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict['schedule'] = self.scheduler.state_dict()
        return state_dict

class Adam(optim.Adam):
    def __init__(self, params, lr, weight_decay, scheduler_option):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        
        name = scheduler_option['scheduler']
        del scheduler_option['scheduler']

        self.scheduler = eval(name)(self, **scheduler_option)
    
    def step(self, closure=None):
        super().step(closure)
        self.scheduler.step()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict['schedule'])
        del state_dict['schedule']
        return super().load_state_dict(state_dict)
    
    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict['schedule'] = self.scheduler.state_dict()
        return state_dict

class AdamW(optim.AdamW):
    def __init__(self, params, lr, weight_decay, scheduler_option):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        
        name = scheduler_option['scheduler']
        del scheduler_option['scheduler']

        self.scheduler = eval(name)(self, **scheduler_option)
    
    def step(self, closure=None):
        super().step(closure)
        self.scheduler.step()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict['schedule'])
        del state_dict['schedule']
        return super().load_state_dict(state_dict)
    
    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict['schedule'] = self.scheduler.state_dict()
        return state_dict