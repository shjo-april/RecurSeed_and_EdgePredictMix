# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import cv2
import sys
import torch

import numpy as np

from torch.nn import functional as F

from tools.ai import torch_utils
from tools.general import io_utils, time_utils

class Evaluator_For_Multi_Label_Classification:
    def __init__(self, class_names, th_interval=0.05):
        self.thresholds = list(np.arange(th_interval, 1.00, th_interval))

        self.class_names = class_names
        self.num_classes = len(self.class_names)
        
        self.clear()

    def clear(self):
        self.meter_dict = {
            th : {
                'P':np.zeros(self.num_classes, dtype=np.float32), 
                'T':np.zeros(self.num_classes, dtype=np.float32), 
                'TP':np.zeros(self.num_classes, dtype=np.float32)
            } for th in self.thresholds}

    def add(self, pred, gt):
        for th in self.thresholds:
            binary_pred = (pred >= th).astype(np.float32)

            self.meter_dict[th]['P'] += binary_pred
            self.meter_dict[th]['T'] += gt
            self.meter_dict[th]['TP'] += (gt * (binary_pred == gt)).astype(np.float32)

    def add_from_data(self, meter_dict):
        for th in self.thresholds:
            self.meter_dict[th]['P'] += meter_dict[th]['P']
            self.meter_dict[th]['T'] += meter_dict[th]['T']
            self.meter_dict[th]['TP'] += meter_dict[th]['TP']

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

class Evaluator_For_Semantic_Segmentation:
    def __init__(self, class_names, ignore_index=255, detail=False):
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.detail = detail
        self.ignore_index = ignore_index

        self.clear()

    def clear(self):
        self.meter_dict = {
            'P' : np.zeros(self.num_classes, dtype=np.float32),
            'T' : np.zeros(self.num_classes, dtype=np.float32),
            'TP' : np.zeros(self.num_classes, dtype=np.float32),

            'FP_BG' : np.zeros(self.num_classes, dtype=np.float32),
            'FP_FG' : np.zeros(self.num_classes, dtype=np.float32),
        }
    
    def add(self, data):
        if isinstance(data, dict):
            pred_mask = torch_utils.get_numpy(data['pred_masks'])[0]
            gt_mask = torch_utils.get_numpy(data['gt_masks'])[0]
        else:
            pred_mask, gt_mask = data
        
        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)

        obj_mask = gt_mask != self.ignore_index
        correct_mask = (pred_mask == gt_mask) * obj_mask

        if self.detail:
            fp_mask = (pred_mask != gt_mask) * obj_mask
            bg_mask = (gt_mask == 0) * obj_mask
            fg_mask = (gt_mask > 0) * obj_mask
        
        for i in range(self.num_classes):
            self.meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
            self.meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
            self.meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)

            if i > 0 and self.detail:
                self.meter_dict['FP_BG'][i] += np.sum((pred_mask==i)*fp_mask*bg_mask)
                self.meter_dict['FP_FG'][i] += np.sum((pred_mask==i)*fp_mask*fg_mask)

    def add_from_data(self, meter_dict):
        for i in range(self.num_classes):
            self.meter_dict['P'][i] += meter_dict['P'][i]
            self.meter_dict['T'][i] += meter_dict['T'][i]
            self.meter_dict['TP'][i] += meter_dict['TP'][i]

            if i > 0 and self.detail:
                self.meter_dict['FP_BG'][i] += meter_dict['FP_BG'][i]
                self.meter_dict['FP_FG'][i] += meter_dict['FP_FG'][i]

    def calculate_per_image(self, pred_mask, gt_mask, label, threshold=None, ignore_index=255, num_classes=21):
        if threshold is not None:
            pred_mask = np.pad(pred_mask, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            pred_mask = np.argmax(pred_mask, axis=0)

        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)
        
        obj_mask = (gt_mask != ignore_index) # | (pred_mask != ignore_index)
        correct_mask = (pred_mask==gt_mask) * obj_mask

        fp_mask = (pred_mask != gt_mask) * obj_mask
        bg_mask = (gt_mask == 0) * obj_mask
        fg_mask = (gt_mask > 0) * obj_mask
        
        IoUs = {}
        FPs = {}
        FNs = {}
        FP_BGs = {}
        FP_FGs = {}
        
        for i in range(num_classes):
            if i > 0:
                if label[i - 1] == 0:
                    continue
            
            P = np.sum((pred_mask==i)*obj_mask)
            T = np.sum((gt_mask==i)*obj_mask)
            TP = np.sum((gt_mask==i)*correct_mask)

            FP_BG = np.sum((pred_mask==i)*fp_mask*bg_mask)
            FP_FG = np.sum((pred_mask==i)*fp_mask*fg_mask)

            # FP_BG_image = (((pred_mask==i)*fp_mask*bg_mask)*255).astype(np.uint8)
            # FP_FG_image = (((pred_mask==i)*fp_mask*fg_mask)*255).astype(np.uint8)

            # cv2.imshow('FP BG', FP_BG_image)
            # cv2.imshow('FP FG', FP_FG_image)
            # cv2.waitKey(0)

            union = (T + P - TP)
            # assert union > 0, 'ZeroDivisionError: {}, {}'.format(image_id, label)
            if union == 0:
                continue

            IoU = TP / union
            FP = (P - TP) / union
            FN = (T - TP) / union

            FP_BG /= union
            FP_FG /= union

            IoUs[self.class_names[i]] = float(IoU)
            IoUs[self.class_names[i]] = float(IoU)
            FPs[self.class_names[i]] = float(FP)
            FNs[self.class_names[i]] = float(FN)

            if i > 0:
                FP_BGs[self.class_names[i]] = float(FP_BG)
                FP_FGs[self.class_names[i]] = float(FP_FG)
        
        return {
            'IoUs': IoUs,

            'FPs': FPs,
            'FNs': FNs,

            'FP_BGs': FP_BGs,
            'FP_FGs': FP_FGs,
        }
    
    def get(self, clear=True, get_class=False):
        IoU_list = []
        FP_list = [] # over activation
        FN_list = [] # under activation

        TP = self.meter_dict['TP']
        P = self.meter_dict['P']
        T = self.meter_dict['T']

        detail_dict = {}
        if self.detail:
            FP_BG = self.meter_dict['FP_BG']
            FP_FG = self.meter_dict['FP_FG']
        
        for i in range(self.num_classes):
            union = (T[i] + P[i] - TP[i])

            IoU = TP[i] / union * 100
            FP = (P[i] - TP[i]) / union
            FN = (T[i] - TP[i]) / union

            detail_dict[self.class_names[i]] = {
                'IoU': IoU,
                'FP': FP,
                'FN': FN,
            }

            if i > 0 and self.detail:
                FP_BG_per_class = FP_BG[i] / union
                FP_FG_per_class = FP_FG[i] / union

                # print(self.class_names[i], IoU, FP, FN, FP_BG_per_class, FP_FG_per_class)

                detail_dict[self.class_names[i]]['FP_BG'] = FP_BG_per_class
                detail_dict[self.class_names[i]]['FP_FG'] = FP_FG_per_class

            print(self.num_classes, self.class_names[i], IoU)

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)
        
        mIoU = np.nanmean(IoU_list)
        # mIoU_foreground = np.nanmean(IoU_list[1:])
        FP = np.nanmean(FP_list[1:])
        FN = np.nanmean(FN_list[1:])
        
        if clear:
            self.clear()
        
        if self.detail:
            return detail_dict
        elif get_class:
            return IoU_list
        else:
            return mIoU, FP, FN
    
    def print(self, tag):
        mIoU, FP, FN = self.get(clear=False)
        print('[{}] mIoU = {:.2f}%, FP = {:.4f}, FN = {:.4f}'.format(tag, mIoU, FP, FN))

        return {
            'mIoU': float(mIoU),
            'FP': float(FP),
            'FN': float(FN)
        }
    
    def print_with_detail(self):
        return self.get(clear=False, get_class=True)

class Evaluator_For_CAM:
    def __init__(self, class_names, ignore_index=255, st_th=0.05, end_th=0.20, th_interval=0.05, detail=False):
        self.thresholds = list(np.arange(st_th, end_th+1e-10, th_interval))

        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.detail = detail
        self.ignore_index = ignore_index

        self.empty = True
        self.clear()

    def clear(self):
        self.meter_dict = {}
        for th in self.thresholds:
            self.meter_dict[th] = {
                'P' : np.zeros(self.num_classes, dtype=np.float32),
                'T' : np.zeros(self.num_classes, dtype=np.float32),
                'TP' : np.zeros(self.num_classes, dtype=np.float32),

                'FP_BG' : np.zeros(self.num_classes, dtype=np.float32),
                'FP_FG' : np.zeros(self.num_classes, dtype=np.float32),
            }
    
    def add(self, data):
        self.empty = False
        pred_cam, gt_mask, label = data

        for th in self.thresholds:
            pred_mask = np.pad(pred_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=th)
            pred_mask = np.argmax(pred_mask, axis=0)

            obj_mask = gt_mask != self.ignore_index
            correct_mask = (pred_mask == gt_mask) * obj_mask

            if self.detail:
                fp_mask = (pred_mask != gt_mask) * obj_mask
                bg_mask = (gt_mask == 0) * obj_mask
                fg_mask = (gt_mask > 0) * obj_mask
            
            for i in range(self.num_classes):
                if i > 0 and label is not None:
                    if label[i - 1] == 0:
                        continue
                
                self.meter_dict[th]['P'][i] += np.sum((pred_mask==i)*obj_mask)
                self.meter_dict[th]['T'][i] += np.sum((gt_mask==i)*obj_mask)
                self.meter_dict[th]['TP'][i] += np.sum((gt_mask==i)*correct_mask)

                if i > 0 and self.detail:
                    self.meter_dict[th]['FP_BG'][i] += np.sum((pred_mask==i)*fp_mask*bg_mask)
                    self.meter_dict[th]['FP_FG'][i] += np.sum((pred_mask==i)*fp_mask*fg_mask)

    def add_from_data(self, pred_cam=None, gt_mask=None, label=None, meter_dict=None):
        self.empty = False
        
        if meter_dict is None:
            # 1. define numpy 
            meter_dict = {}
            for th in self.thresholds:
                meter_dict[th] = {
                    'P' : np.zeros(self.num_classes, dtype=np.float32),
                    'T' : np.zeros(self.num_classes, dtype=np.float32),
                    'TP' : np.zeros(self.num_classes, dtype=np.float32),

                    'FP_BG' : np.zeros(self.num_classes, dtype=np.float32),
                    'FP_FG' : np.zeros(self.num_classes, dtype=np.float32),
                }
            
            # 2. calculate P, T, and TP
            for th in self.thresholds:
                pred_mask = np.pad(pred_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=th)
                pred_mask = np.argmax(pred_mask, axis=0)

                obj_mask = gt_mask != self.ignore_index
                correct_mask = (pred_mask==gt_mask) * obj_mask

                if self.detail:
                    fp_mask = (pred_mask != gt_mask) * obj_mask
                    bg_mask = (gt_mask == 0) * obj_mask
                    fg_mask = (gt_mask > 0) * obj_mask
                
                for i in range(self.num_classes):
                    if i > 0:
                        if label[i - 1] == 0:
                            continue
                    
                    meter_dict[th]['P'][i] += np.sum((pred_mask==i)*obj_mask)
                    meter_dict[th]['T'][i] += np.sum((gt_mask==i)*obj_mask)
                    meter_dict[th]['TP'][i] += np.sum((gt_mask==i)*correct_mask)

                    if i > 0 and self.detail:
                        meter_dict[th]['FP_BG'][i] += np.sum((pred_mask==i)*fp_mask*bg_mask)
                        meter_dict[th]['FP_FG'][i] += np.sum((pred_mask==i)*fp_mask*fg_mask)

            # print('# {}ms'.format(self.timer.tok(ms=True, clear=True)))
        
        # 3. update P, T, and TP
        for th in self.thresholds:
            for i in range(self.num_classes):
                if i > 0 and label is not None:
                    if label[i - 1] == 0:
                        continue

                self.meter_dict[th]['P'][i] += meter_dict[th]['P'][i]
                self.meter_dict[th]['T'][i] += meter_dict[th]['T'][i]
                self.meter_dict[th]['TP'][i] += meter_dict[th]['TP'][i]

                if i > 0 and self.detail:
                    self.meter_dict[th]['FP_BG'][i] += meter_dict[th]['FP_BG'][i]
                    self.meter_dict[th]['FP_FG'][i] += meter_dict[th]['FP_FG'][i]

    def get(self, clear=True):
        mIoU_list = []
        FP_list = []
        FN_list = []

        detail_dict_list = []

        for th in self.thresholds:
            _IoU_list = []
            _FP_list = [] # over activation
            _FN_list = [] # under activation

            TP = self.meter_dict[th]['TP']
            P = self.meter_dict[th]['P']
            T = self.meter_dict[th]['T']

            detail_dict = {}
            if self.detail:
                FP_BG = self.meter_dict[th]['FP_BG']
                FP_FG = self.meter_dict[th]['FP_FG']
            
            for i in range(self.num_classes):
                union = (T[i] + P[i] - TP[i])

                IoU = TP[i] / union * 100
                FP = (P[i] - TP[i]) / union
                FN = (T[i] - TP[i]) / union

                detail_dict[self.class_names[i]] = {
                    'IoU': IoU,
                    'FP': FP,
                    'FN': FN,
                }

                if i > 0 and self.detail:
                    FP_BG_per_class = FP_BG[i] / union
                    FP_FG_per_class = FP_FG[i] / union

                    # print(self.class_names[i], IoU, FP, FN, FP_BG_per_class, FP_FG_per_class)

                    detail_dict[self.class_names[i]]['FP_BG'] = FP_BG_per_class
                    detail_dict[self.class_names[i]]['FP_FG'] = FP_FG_per_class

                _IoU_list.append(IoU)
                _FP_list.append(FP)
                _FN_list.append(FN)
            
            mIoU = np.nanmean(_IoU_list)
            # mIoU_foreground = np.nanmean(IoU_list[1:])
            FP = np.nanmean(_FP_list[1:])
            FN = np.nanmean(_FN_list[1:])

            mIoU_list.append(mIoU)
            FP_list.append(FP)
            FN_list.append(FN)
            detail_dict_list.append(detail_dict)
        
        if clear:
            self.clear()
        
        if self.detail:
            return self.thresholds, mIoU_list, FP_list, FN_list, detail_dict_list
        else:
            best_index = np.argmax(mIoU_list)
            best_th = self.thresholds[best_index]

            best_FP = FP_list[best_index]
            best_FN = FN_list[best_index]

            best_mIoU = mIoU_list[best_index]

            return best_th, best_mIoU, best_FP, best_FN
    
    def print(self, tag):
        if self.detail:
            return self.get(clear=False)
        else:
            th, mIoU, FP, FN = self.get(clear=False)
            print('[{}] Threshold = {:.2f}, mIoU = {:.2f}%, FP = {:.4f}, FN = {:.4f}'.format(tag, th, mIoU, FP, FN))

            return {
                'threshold': round(float(th), 2),
                'mIoU': float(mIoU),
                'FP': float(FP),
                'FN': float(FN)
            }
