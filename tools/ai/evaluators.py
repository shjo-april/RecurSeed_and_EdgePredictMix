# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import sys
import torch

import numpy as np

from torch.nn import functional as F

from tools.ai import torch_utils, demo_utils
from tools.general import io_utils, time_utils

class Guide_For_Evaluation:
    def __init__(self, model, loader, trainer, class_names, amp, ema, num_samples):
        self.model = model
        self.loader = loader

        self.eval_timer = time_utils.Timer()
        self.num_iterations = len(self.loader)

        self.amp = amp
        self.ema = ema
        self.trainer = trainer

        self.num_samples = num_samples
        self.class_names = class_names

    def inference(self, images, labels):
        raise NotImplementedError

    def update(self, data_dict):
        raise NotImplementedError

    def upload(self, data_dict):
        raise NotImplementedError

    def return_results(self):
        raise NotImplementedError

    def preprocess(self, images, labels):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        return images, labels
    
    def step(self, debug=False):
        self.model.eval()
        self.eval_timer.tik()
        
        ni_digits = io_utils.get_digits_in_number(self.num_iterations)
        progress_format = '\r# Evaluation [%0{}d/%0{}d] = %02.2f%%'.format(ni_digits, ni_digits)
        
        for i, (_, images, labels, gt_masks) in enumerate(self.loader):
            i += 1
            
            # preprocess
            images, labels = self.preprocess(images, labels)
            
            # infer
            data_dict = self.inference(images, labels)

            # update
            data_dict['number'] = i
            data_dict['images'] = images
            data_dict['labels'] = labels
            data_dict['gt_masks'] = gt_masks

            self.update(data_dict)

            if i <= self.num_samples:
                self.upload(data_dict)
            
            sys.stdout.write(progress_format%(i, self.num_iterations, i / self.num_iterations * 100))
            sys.stdout.flush()

            if debug:
                break
        
        print('\r', end='')
        self.model.train()

        return self.return_results(), self.eval_timer.tok(clear=True)

class Guide_For_Segmentation:
    def __init__(self, model, loader, trainer, class_names, amp, ema, num_samples):
        self.model = model
        self.loader = loader

        self.eval_timer = time_utils.Timer()
        self.num_iterations = len(self.loader)

        self.amp = amp
        self.ema = ema
        self.trainer = trainer

        self.num_samples = num_samples
        self.class_names = class_names

    def inference(self, images):
        raise NotImplementedError

    def update(self, data_dict):
        raise NotImplementedError

    def upload(self, data_dict):
        raise NotImplementedError

    def return_results(self):
        raise NotImplementedError

    def preprocess(self, images):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        return images
    
    def step(self, debug=False):
        self.model.eval()
        self.eval_timer.tik()
        
        ni_digits = io_utils.get_digits_in_number(self.num_iterations)
        progress_format = '\r# Evaluation [%0{}d/%0{}d] = %02.2f%%'.format(ni_digits, ni_digits)
        
        for i, (images, gt_masks) in enumerate(self.loader):
            i += 1
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            
            # preprocess
            images = self.preprocess(images)
            
            # infer
            data_dict = self.inference(images)

            # update
            data_dict['number'] = i
            data_dict['images'] = images
            data_dict['gt_masks'] = gt_masks

            self.update(data_dict)

            if i <= self.num_samples:
                self.upload(data_dict)
            
            sys.stdout.write(progress_format%(i, self.num_iterations, i / self.num_iterations * 100))
            sys.stdout.flush()

            if debug:
                break
        
        print('\r', end='')
        self.model.train()

        return self.return_results(), self.eval_timer.tok(clear=True)

class Evaluator_For_Multi_Label_Classification:
    def __init__(self, class_names, th_interval=0.05):
        self.thresholds = list(np.arange(th_interval, 1.00, th_interval))

        self.class_names = class_names
        self.num_classes = len(self.class_names)
        
        self.clear()

    def add(self, pred, gt):
        for th in self.thresholds:
            binary_pred = (pred >= th).astype(np.float32)

            self.meter_dict[th]['P'] += binary_pred
            self.meter_dict[th]['T'] += gt
            self.meter_dict[th]['TP'] += (gt * (binary_pred == gt)).astype(np.float32)

    def get(self, detail=False, clear=True):
        op_list = []
        or_list = []
        o_f1_list = []

        cp_list = []
        cr_list = []
        c_f1_list = []

        for th in self.thresholds:
            data = self.meter_dict[th]

            P = data['P']
            T = data['T']
            TP = data['TP']

            overall_precision = np.sum(TP) / (np.sum(P) + 1e-5) * 100
            overall_recall = np.sum(TP) / (np.sum(T) + 1e-5) * 100
            overall_f1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-5))

            op_list.append(overall_precision)
            or_list.append(overall_recall)
            o_f1_list.append(overall_f1_score)

            per_class_precision = np.mean(TP / (P + 1e-5)) * 100
            per_class_recall = np.mean(TP / (T + 1e-5)) * 100
            per_class_f1_score = 2 * ((per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + 1e-5))

            cp_list.append(per_class_precision)
            cr_list.append(per_class_recall)
            c_f1_list.append(per_class_f1_score)

        # best_index = np.argmax(o_f1_list)
        # best_o_th = self.thresholds[best_index]

        # best_op = op_list[best_index]
        # best_or = or_list[best_index]
        # best_of = o_f1_list[best_index]

        best_index = np.argmax(c_f1_list)
        best_c_th = self.thresholds[best_index]
        
        best_cp = cp_list[best_index]
        best_cr = cr_list[best_index]
        best_cf = c_f1_list[best_index]

        if clear:
            self.clear()
        
        if detail:
            return [best_c_th, best_cp, best_cr, best_cf], cp_list, cr_list, c_f1_list
        else:
            return best_c_th, best_cf
    
    def clear(self):
        self.meter_dict = {
            th : {
                'P':np.zeros(self.num_classes, dtype=np.float32), 
                'T':np.zeros(self.num_classes, dtype=np.float32), 
                'TP':np.zeros(self.num_classes, dtype=np.float32)
            } for th in self.thresholds}

class Evaluator_For_Semantic_Segmentation:
    def __init__(self, class_names, ignore_index=255):
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.ignore_index = ignore_index

        self.clear()

    def clear(self):
        self.meter_dict = {
            'P' : np.zeros(self.num_classes, dtype=np.float32),
            'T' : np.zeros(self.num_classes, dtype=np.float32),
            'TP' : np.zeros(self.num_classes, dtype=np.float32)
        }
    
    def add(self, pred_mask, gt_mask):
        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)

        obj_mask = gt_mask != self.ignore_index
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        for i in range(self.num_classes):
            self.meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
            self.meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
            self.meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)
    
    def get(self, detail=False, clear=True):
        IoU_dict = {}
        IoU_list = []

        FP_list = [] # over activation
        FN_list = [] # under activation

        TP = self.meter_dict['TP']
        P = self.meter_dict['P']
        T = self.meter_dict['T']
        
        for i in range(self.num_classes):
            IoU = TP[i] / (T[i] + P[i] - TP[i]) * 100
            FP = (P[i] - TP[i]) / (T[i] + P[i] - TP[i])
            FN = (T[i] - TP[i]) / (T[i] + P[i] - TP[i])

            IoU_dict[self.class_names[i]] = IoU
            IoU_list.append(IoU)

            FP_list.append(FP)
            FN_list.append(FN)
        
        mIoU = np.nanmean(IoU_list)
        mIoU_foreground = np.nanmean(IoU_list[1:])
        
        FP = np.nanmean(FP_list[1:])
        FN = np.nanmean(FN_list[1:])
        
        if clear:
            self.clear()
        
        if detail:
            return mIoU, mIoU_foreground, IoU_dict, FP, FN
        else:
            return mIoU, FP, FN
    
    def print(self, tag):
        mIoU, FP, FN = self.get(detail=False, clear=False)
        print('[{}] mIoU = {:.2f}%, FP = {:.4f}, FN = {:.4f}'.format(tag, mIoU, FP, FN))

        return {
            'mIoU': float(mIoU),
            'FP': float(FP),
            'FN': float(FN)
        }
    
    def print_with_detail(self):
        mIoU, _, IoU_dict, FP, FN = self.get(detail=True, clear=False)
        return IoU_dict

    @staticmethod
    def calculate_mIoU_per_image(pred_mask, gt_mask, label, threshold=None, ignore_index=255, num_classes=21, mean=True):
        P = np.zeros(num_classes, dtype=np.float32)
        T = np.zeros(num_classes, dtype=np.float32)
        TP = np.zeros(num_classes, dtype=np.float32)
        
        if threshold is not None:
            pred_mask = np.pad(pred_mask, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            pred_mask = np.argmax(pred_mask, axis=0)
        
        obj_mask = (gt_mask != ignore_index) | (pred_mask != ignore_index)
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        IoUs = []
        
        for i in range(num_classes):
            if i > 0:
                if label[i - 1] == 0:
                    continue
            
            P[i] += np.sum((pred_mask==i)*obj_mask)
            T[i] += np.sum((gt_mask==i)*obj_mask)
            TP[i] += np.sum((gt_mask==i)*correct_mask)

            IoU = TP[i] / (T[i] + P[i] - TP[i])
            IoUs.append(IoU)
        
        if mean:
            mIoU = float(np.nanmean(IoUs) * 100)
            return mIoU
        return IoUs

    @staticmethod
    def calculate_details_per_image(pred_mask, gt_mask, label, threshold=None, ignore_index=255, num_classes=21):
        P = np.zeros(num_classes, dtype=np.float32)
        T = np.zeros(num_classes, dtype=np.float32)
        TP = np.zeros(num_classes, dtype=np.float32)
        
        if threshold is not None:
            pred_mask = np.pad(pred_mask, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            pred_mask = np.argmax(pred_mask, axis=0)
        
        obj_mask = (gt_mask != ignore_index) | (pred_mask != ignore_index)
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        IoUs = []
        FPs = []
        FNs = []
        
        for i in range(num_classes):
            if i > 0:
                if label[i - 1] == 0:
                    continue
            
            P[i] += np.sum((pred_mask==i)*obj_mask)
            T[i] += np.sum((gt_mask==i)*obj_mask)
            TP[i] += np.sum((gt_mask==i)*correct_mask)

            IoU = TP[i] / (T[i] + P[i] - TP[i])
            FP = (P[i] - TP[i]) / (T[i] + P[i] - TP[i])
            FN = (T[i] - TP[i]) / (T[i] + P[i] - TP[i])

            IoUs.append(float(IoU))
            FPs.append(float(FP))
            FNs.append(float(FN))
        
        return IoUs, FPs, FNs

class Evaluator_For_CAM:
    def __init__(self, class_names, ignore_index=255, st_th=0.05, end_th=0.15, th_interval=0.05):
        self.thresholds = list(np.arange(st_th, end_th+1e-10, th_interval))

        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.ignore_index = ignore_index

        self.clear()

        self.timer = time_utils.Timer()

    def clear(self):
        self.meter_dict = {}
        for th in self.thresholds:
            self.meter_dict[th] = {
                'P' : np.zeros(self.num_classes, dtype=np.float32),
                'T' : np.zeros(self.num_classes, dtype=np.float32),
                'TP' : np.zeros(self.num_classes, dtype=np.float32)
            }
    
    def add(self, pred_cam, gt_mask, label):
        for th in self.thresholds:
            pred_mask = np.pad(pred_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=th)
            pred_mask = np.argmax(pred_mask, axis=0)

            obj_mask = gt_mask != self.ignore_index
            correct_mask = (pred_mask==gt_mask) * obj_mask
            
            for i in range(self.num_classes):
                if i > 0:
                    if label[i - 1] == 0:
                        continue

                self.meter_dict[th]['P'][i] += np.sum((pred_mask==i)*obj_mask)
                self.meter_dict[th]['T'][i] += np.sum((gt_mask==i)*obj_mask)
                self.meter_dict[th]['TP'][i] += np.sum((gt_mask==i)*correct_mask)
    
    def add_from_data(self, pred_cam=None, gt_mask=None, label=None, meter_dict=None):
        if meter_dict is None:
            # 1. define numpy 
            meter_dict = {}
            for th in self.thresholds:
                meter_dict[th] = {
                    'P' : np.zeros(self.num_classes, dtype=np.float32),
                    'T' : np.zeros(self.num_classes, dtype=np.float32),
                    'TP' : np.zeros(self.num_classes, dtype=np.float32)
                }
            
            # 2. calculate P, T, and TP
            self.timer.tik()

            for th in self.thresholds:
                pred_mask = np.pad(pred_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=th)
                pred_mask = np.argmax(pred_mask, axis=0)

                obj_mask = gt_mask != self.ignore_index
                correct_mask = (pred_mask==gt_mask) * obj_mask
                
                for i in range(self.num_classes):
                    if i > 0:
                        if label[i - 1] == 0:
                            continue
                    
                    meter_dict[th]['P'][i] += np.sum((pred_mask==i)*obj_mask)
                    meter_dict[th]['T'][i] += np.sum((gt_mask==i)*obj_mask)
                    meter_dict[th]['TP'][i] += np.sum((gt_mask==i)*correct_mask)

            # print('# {}ms'.format(self.timer.tok(ms=True, clear=True)))
        
        # 3. update P, T, and TP
        for th in self.thresholds:
            for i in range(self.num_classes):
                self.meter_dict[th]['P'][i] += meter_dict[th]['P'][i]
                self.meter_dict[th]['T'][i] += meter_dict[th]['T'][i]
                self.meter_dict[th]['TP'][i] += meter_dict[th]['TP'][i]

    def get(self, detail=False, clear=True, foreground=False):
        mIoU_list = []
        mIoU_foreground_list = []

        IoU_dict_list = []
        FP_list = []
        FN_list = []

        for th in self.thresholds:
            _IoU_dict = {}
            _IoU_list = []

            _FP_list = [] # over activation
            _FN_list = [] # under activation

            TP = self.meter_dict[th]['TP']
            P = self.meter_dict[th]['P']
            T = self.meter_dict[th]['T']

            for i in range(self.num_classes):
                IoU = TP[i] / (T[i] + P[i] - TP[i]) * 100
                FP = (P[i] - TP[i]) / (T[i] + P[i] - TP[i])
                FN = (T[i] - TP[i]) / (T[i] + P[i] - TP[i])

                _IoU_dict[self.class_names[i]] = IoU
                _IoU_list.append(IoU)

                _FP_list.append(FP)
                _FN_list.append(FN)
            
            mIoU = np.nanmean(_IoU_list)
            mIoU_foreground = np.nanmean(_IoU_list[1:])
            
            FP = np.nanmean(_FP_list[1:])
            FN = np.nanmean(_FN_list[1:])
            
            mIoU_list.append(mIoU)
            mIoU_foreground_list.append(mIoU_foreground)

            IoU_dict_list.append(_IoU_dict) 

            FP_list.append(FP)
            FN_list.append(FN)

        if foreground:
            mIoU_list = mIoU_foreground_list
        
        if clear:
            self.clear()

        if detail:
            return self.thresholds, mIoU_list, IoU_dict_list, FP_list, FN_list
        else:
            best_index = np.argmax(mIoU_list)
            best_th = self.thresholds[best_index]

            best_FP = FP_list[best_index]
            best_FN = FN_list[best_index]

            best_mIoU = mIoU_list[best_index]

            return best_th, best_mIoU, best_FP, best_FN
    
    def print_with_detail(self):
        th_list, mIoU_list, _, FP_list, FN_list = self.get(detail=True, clear=False)
        
        detail_dict = {}
        for th, mIoU, FP, FN in zip(th_list, mIoU_list, FP_list, FN_list):
            detail_dict['{:.2f}'.format(th)] = {
                'mIoU': float(mIoU),
                'FP': float(FP),
                'FN': float(FN)
            }

        return detail_dict

    def print(self, tag, foreground=False):
        th, mIoU, FP, FN = self.get(detail=False, clear=False, foreground=foreground)
        print('[{}] Threshold = {:.2f}, mIoU = {:.2f}%, FP = {:.4f}, FN = {:.4f}'.format(tag, th, mIoU, FP, FN))

        return {
            'threshold': round(float(th), 2),
            'mIoU': float(mIoU),
            'FP': float(FP),
            'FN': float(FN)
        }

    @staticmethod
    def calculate_mIoU_per_image(pred_mask, gt_mask, label, ignore_index=255, num_classes=21, mean=True, thresholds=None):
        if thresholds is None:
            thresholds = list(np.arange(0.10, 0.35+1e-10, 0.05))
        
        mIoUs = []
        for th in thresholds:
            mIoU = Evaluator_For_Semantic_Segmentation.calculate_mIoU_per_image(pred_mask, gt_mask, label, th, ignore_index, num_classes, mean)
            mIoUs.append(mIoU)
        return np.max(mIoUs)

    @staticmethod
    def calculate_details_per_image(pred_mask, gt_mask, label, ignore_index=255, num_classes=21, thresholds=None):
        if thresholds is None:
            thresholds = list(np.arange(0.10, 0.35+1e-10, 0.05))
        
        mIoUs = []
        IoUs = []
        FPs = []
        FNs = []

        for th in thresholds:
            IoU, FP, FN = Evaluator_For_Semantic_Segmentation.calculate_details_per_image(pred_mask, gt_mask, label, th, ignore_index, num_classes)

            mIoUs.append(np.mean(IoU))
            IoUs.append(IoU)
            FPs.append(FP)
            FNs.append(FN)
        
        # print(mIoUs, np.argmax(mIoUs))

        index = np.argmax(mIoUs)
        return IoUs[index][1:], FPs[index][1:], FNs[index][1:]