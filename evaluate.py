# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import sys
import cv2
import ray

import tqdm
import torch
import numpy as np

from PIL import Image

from core import datasets

from tools.ai import evaluators, augment_utils
from tools.general import io_utils, json_utils, cv_utils, pickle_utils

@ray.remote
def update_mIoU(obj, pred_cam, gt_mask, label):
    # 1. define numpy 
    meter_dict = {}
    for th in obj.thresholds:
        meter_dict[th] = {
            'P' : np.zeros(obj.num_classes, dtype=np.float32),
            'T' : np.zeros(obj.num_classes, dtype=np.float32),
            'TP' : np.zeros(obj.num_classes, dtype=np.float32),

            'FP_BG' : np.zeros(obj.num_classes, dtype=np.float32),
            'FP_FG' : np.zeros(obj.num_classes, dtype=np.float32),
        }
    
    # 2. calculate P, T, and TP
    for th in obj.thresholds:
        pred_mask = np.pad(pred_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=th)
        pred_mask = np.argmax(pred_mask, axis=0)

        obj_mask = gt_mask != obj.ignore_index
        correct_mask = (pred_mask==gt_mask) * obj_mask

        if obj.detail:
            fp_mask = (pred_mask != gt_mask) * obj_mask
            bg_mask = (gt_mask == 0) * obj_mask
            fg_mask = (gt_mask > 0) * obj_mask
        
        for i in range(obj.num_classes):
            if i > 0 and label is not None:
                if label[i - 1] == 0:
                    continue
            
            meter_dict[th]['P'][i] += np.sum((pred_mask==i)*obj_mask)
            meter_dict[th]['T'][i] += np.sum((gt_mask==i)*obj_mask)
            meter_dict[th]['TP'][i] += np.sum((gt_mask==i)*correct_mask)

            if i > 0 and obj.detail:
                meter_dict[th]['FP_BG'][i] += np.sum((pred_mask==i)*fp_mask*bg_mask)
                meter_dict[th]['FP_FG'][i] += np.sum((pred_mask==i)*fp_mask*fg_mask)

    meter_dict['label'] = label

    return meter_dict

@ray.remote
def update_mIoU_for_semantic_segmentation(obj, pred_mask, gt_mask):
    # 1. define numpy 
    meter_dict = {
        'P' : np.zeros(obj.num_classes, dtype=np.float32),
        'T' : np.zeros(obj.num_classes, dtype=np.float32),
        'TP' : np.zeros(obj.num_classes, dtype=np.float32),

        'FP_BG' : np.zeros(obj.num_classes, dtype=np.float32),
        'FP_FG' : np.zeros(obj.num_classes, dtype=np.float32),
    }
    
    if len(pred_mask.shape) == 3:
        pred_mask = np.argmax(pred_mask, axis=0)

    obj_mask = gt_mask != obj.ignore_index
    correct_mask = (pred_mask == gt_mask) * obj_mask

    if obj.detail:
        fp_mask = (pred_mask != gt_mask) * obj_mask
        bg_mask = (gt_mask == 0) * obj_mask
        fg_mask = (gt_mask > 0) * obj_mask
    
    for i in range(obj.num_classes):
        meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
        meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
        meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)

        if i > 0 and obj.detail:
            meter_dict['FP_BG'][i] += np.sum((pred_mask==i)*fp_mask*bg_mask)
            meter_dict['FP_FG'][i] += np.sum((pred_mask==i)*fp_mask*fg_mask)

    return meter_dict

def main(args):
    # set directories
    pred_dir = args.pred_dir + f'{args.folder}/{args.tag}/{args.domain}/'

    if args.detail:
        analysis_dir = io_utils.create_directory('./experiments/analysis/per-class/')
        analysis_path = analysis_dir + args.tag + '.json'

    log_func = lambda string='': print(string)

    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    # for datasets
    test_transform = augment_utils.Compose(
        [
            augment_utils.Normalize(),
            augment_utils.Transpose(),
        ]
    )

    test_dataset = datasets.Dataset_For_Analysis(
        args.root_dir, 
        args.domain, 
        test_transform,
        name=args.dataset,
        single=args.single
    )

    # for evaluation
    colors = cv_utils.get_colors(data_dict)
    denorm_fn = augment_utils.Denormalize()

    if args.parallel or 'predictions' in args.folder:
        ids = []
        ray.init(num_cpus=args.num_workers)

    if 'predictions' in args.folder:
        evaluator_for_cam = evaluators.Evaluator_For_CAM(data_dict['class_names'], ignore_index=255, st_th=args.st_th, end_th=args.end_th, th_interval=args.th_interval)
    else:
        evaluator_for_ss = evaluators.Evaluator_For_Semantic_Segmentation(data_dict['class_names'], ignore_index=255)

        print(evaluator_for_ss.class_names)
    
    if args.debug:
        count = 0

    for image_id, image, label, gt_mask in tqdm.tqdm(test_dataset):
        gt_mask = np.asarray(gt_mask)

        if args.debug:
            count += 1
            if count >= 2500:
                break

            # if not os.path.isfile(pred_dir + image_id + '.pkl'):
            #     continue
        
        if 'predictions' in args.folder:
            infer_dict = pickle_utils.load_pickle(pred_dir + image_id + '.pkl')

            # print()
            # print(infer_dict['keys'][1:]-1)
            # print(infer_dict['seed'].shape)
            # print()

            _, h, w = image.shape
            pred = np.zeros((len(label), h, w), dtype=np.float32)
            if len(infer_dict['keys'][1:]) > 0:
                pred[infer_dict['keys'][1:]-1] = infer_dict['seed']
            
            """
            # evaluator_for_cam.add([pred, gt_mask, label])
            evaluator_for_cam.add([pred, gt_mask, None])
            """

            id = update_mIoU.remote(
                evaluator_for_cam, 
                pred, 
                gt_mask,
                label if args.domain == 'train' else None
            )
            ids.append(id)

            if len(ids) > (args.num_workers * 10):
                for meter_dict in ray.get(ids):
                    evaluator_for_cam.add_from_data(
                        label=meter_dict['label'],
                        meter_dict=meter_dict
                    )
                ids = []
        else:
            pred_mask = Image.open(pred_dir + image_id + '.png')
            pred_mask = np.asarray(pred_mask)

            # print(image_id)

            if args.parallel:
                id = update_mIoU_for_semantic_segmentation.remote(
                    evaluator_for_ss, 
                    pred_mask, 
                    gt_mask,
                )
                ids.append(id)

                # print(len(ids), args.num_workers * 10)

                if len(ids) > (args.num_workers * 10):
                    for meter_dict in ray.get(ids):
                        evaluator_for_ss.add_from_data(meter_dict)

                    # print(evaluator_for_ss.meter_dict['P'][0])
                    # print(evaluator_for_ss.meter_dict['T'][0])
                    # print(evaluator_for_ss.meter_dict['TP'][0])
                    # print()

                    ids = []
            else:
                evaluator_for_ss.add([pred_mask, gt_mask])
        
        # if args.debug:
        #     cv2.imshow('Ground Truth', colors[gt_mask])
        #     cv2.imshow('Prediction', cv_utils.apply_colormap(pred))
        #     # cv2.imshow('Prediction', colors[pred_mask])
        #     cv2.waitKey(0)

    if 'predictions' in args.folder:
        if len(ids) > 0:
            for meter_dict in ray.get(ids):
                evaluator_for_cam.add_from_data(
                    label=meter_dict['label'],
                    meter_dict=meter_dict
                )
        
        evaluator_for_cam.print(args.tag)
    else:
        if args.parallel and len(ids) > 0:
            for meter_dict in ray.get(ids):
                evaluator_for_ss.add_from_data(meter_dict)
        
        if args.detail:
            IoU_list = evaluator_for_ss.print_with_detail()
            class_names = evaluator_for_ss.class_names

            print(len(class_names), len(IoU_list), np.mean(IoU_list))

            IoU_dict = {
                name: float(IoU) for IoU, name in zip(IoU_list, class_names)
            }
            IoU_dict['mIoU'] = np.mean(IoU_list)
            json_utils.write_json(analysis_path, IoU_dict)
        else:
            evaluator_for_ss.print(args.tag)

if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # environment
    parser.add('gpus', '0', str)
    parser.add('num_workers', 16, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('pred_dir', './experiments/', str)

    parser.add('domain', 'train', str)
    parser.add('single', False, bool)
    
    # evaluation configuration
    parser.add('tag', 'ResNet50@RS(ep=5, cam=1.0, seg=1.0)+CF(c=0.55, u=0.10)+CM(ep=15, cls=1.0, cam=1.0, seg=1.0)', str)
    parser.add('folder', 'pseudo-labels', str)

    parser.add('debug', False, bool)
    parser.add('parallel', False, bool)
    parser.add('detail', False, bool)

    parser.add('st_th', 0.10, float)
    parser.add('end_th', 0.80, float)
    parser.add('th_interval', 0.05, float)
    
    main(parser.get_args())
