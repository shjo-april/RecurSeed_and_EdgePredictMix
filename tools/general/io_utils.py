# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import socket
import argparse

import numpy as np

def get_hostname():
    return socket.gethostname()

def get_digits_in_number(number):
    count = 0
    while number > 0:
        count += 1
        number //= 10
    return count

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def boolean(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def list_type(v):
    return list(map(float, v.split()))

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add(self, tag, default, type):
        if type == bool:
            type = boolean
        
        # if type == list:
        #     type = list_type
        
        self.parser.add_argument(f'--{tag}', default=default, type=type)

    def add_from_list(self, dataset):
        for data in dataset:
            self.add(*data)

    def add_from_dict(self, dataset):
        for tag in dataset.keys():
            default = dataset[tag]['default']
            type = dataset[tag]['type']

            self.add(tag, default, type)
    
    def get_args(self):
        return self.parser.parse_args()
