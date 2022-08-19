# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import time
import datetime

def convert_sec2time(seconds):
    minutes = seconds // 60
    hours = minutes // 60

    minutes = minutes % 60
    seconds = seconds % 60

    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

        self.tik()
    
    def tik(self):
        self.start_time = time.time()
    
    def tok(self, ms: bool=False, clear: bool=False):
        self.end_time = time.time()
        
        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.tik()

        return duration