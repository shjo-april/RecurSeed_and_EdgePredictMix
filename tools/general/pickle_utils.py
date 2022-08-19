# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import pickle

def dump_pickle(path, data):
    pickle.dump(data, open(path, 'wb'))

def load_pickle(path):
    return pickle.load(open(path, 'rb'))
