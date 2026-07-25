[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recurseed-and-certainmix-for-weakly/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=recurseed-and-certainmix-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recurseed-and-certainmix-for-weakly/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=recurseed-and-certainmix-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recurseed-and-certainmix-for-weakly/weakly-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-4?p=recurseed-and-certainmix-for-weakly)

# RecurSeed and EdgePredictMix
This repository is the official implementation of "RecurSeed and EdgePredictMix: Single-stage learning is sufficient for Weakly-Supervised Semantic Segmentation". Please feel free to reach out for any questions or discussions.

> 📢 **News:** Our follow-up work, [MARS: Model-agnostic Biased Object Removal without Additional Supervision for Weakly-Supervised Semantic Segmentation](https://github.com/shjo-april/MARS) (ICCV 2023), builds on RS+EPM and is now publicly available with its complete implementation.

# Citation
- Please cite our paper if the code is helpful to your research. [arxiv](https://arxiv.org/abs/2204.06754)
```
@article{jo2022recurseed,
  title={RecurSeed and EdgePredictMix: Single-stage Learning is Sufficient for Weakly-Supervised Semantic Segmentation},
  author={Jo, Sanghyun and Yu, In-Jae and Kim, Kyungsu},
  journal={arXiv preprint arXiv:2204.06754},
  year={2022}
}
```

### Abstract
Although weakly-supervised semantic segmentation using only image-level labels (WSSS-IL) is potentially useful, its low performance and implementation complexity still limit its application. The main causes are (a) non-detection and (b) false-detection phenomena: (a) The class activation maps refined from existing WSSS-IL methods still only represent partial regions for large-scale objects, and (b) for small-scale objects, over-activation causes them to deviate from the object edges. We propose RecurSeed which alternately reduces non- and false detections through recursive iterations, thereby implicitly finding an optimal junction that minimizes both errors. We also propose a novel data augmentation (DA) approach called EdgePredictMix, which further expresses an object's edge by utilizing the probability difference information between adjacent pixels in combining the segmentation results, thereby compensating for the shortcomings when applying the existing DA methods to WSSS. We achieved new state-of-the-art performances on both the PASCAL VOC 2012 and MS COCO 2014 benchmarks (VOC \emph{val} $74.4\%$, COCO \emph{val} $46.4\%$).

![Overview](./resources/Overview.jpg)

# Update

[07/25/2026] Released the complete implementation of RS+EPM (training, inference, and evaluation). We sincerely apologize for the long delay in releasing the code, which was due to company commitments.

[01/26/2023] Released additional weights trained on the COCO 2014 dataset and updating citation.

[10/18/2022] Released additional results such as pseudo labels and DeepLabv2.

[08/19/2022] Released "RecurSeed and EdgePredictMix" for testing.

[04/14/2022] Released "RecurSeed and CertainMix" for testing.

# Setup

Setting up for this project involves installing dependencies and preparing datasets. The code is tested on Ubuntu 20.04 with NVIDIA GPUs and CUDA installed. 

### Installing dependencies
To install all dependencies, please run the following:
```bash
python3 -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git
python3 -m pip install -r requirements.txt
```

### Preparing datasets

Please download following VOC and COCO datasets. Each dataset has a different directory structure. Therefore, we modify directory structures of all datasets for a comfortable implementation. 

> ##### 1. PASCAL VOC 2012
> Download PASCAL VOC 2012 dataset from our [[Google Drive](https://drive.google.com/file/d/1ITnF19LayDdC1QYUki1guta82L9qSrGo/view?usp=sharing)].

> ##### 2. MS COCO 2014
> Download MS COCO 2014 dataset from our [[Google Drive](https://drive.google.com/file/d/1WwcK-33wHpGw4cozEi7hkmpI0GuYoPEc/view?usp=sharing)].

Create a directory "../VOC2012/" for storing the dataset and appropriately place each dataset to have the following directory structure.
```
    ../                               # parent directory
    ├── ./                            # current (project) directory
    │   ├── core/                     # (dir.) implementation of RecurSeed and EdgePredictMix
    │   ├── data/                     # (dir.) information per dataset (including class names and the number of classes)
    │   ├── tools/                    # (dir.) helper functions
    │   ├── README.md                 # intstruction for a reproduction
    │   └── ... some python files ...
    |
    ├── VOC2012/                      # PASCAL VOC 2012
    │   ├── train/              
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/        
    │   ├── train_aug/
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/   
    │   ├── validation/
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/   
    │   └── test/
    │       └── image/
    |
    └── COCO2014/                     # MS COCO 2014
        ├── train/              
        │   ├── image/     
        │   ├── mask/        
        │   └── xml/
        └── validation/
            ├── image/     
            ├── mask/        
            └── xml/
```

# Visualization
We prepared [a jupyter notebook](https://github.com/shjo-april/RecurSeed_and_EdgePredictMix/blob/master/demo.ipynb) for visualization.

# Training
Below lines are training commands to reproduce our method.

### 1. RecurSeed and EdgePredictMix (single-stage)
Training takes approx. 24 hours until convergence (trained with a single RTX A6000 GPU).

```bash
# Train our single-stage method with RecurSeed and EdgePredictMix.

# 1. PASCAL VOC 2012
python3 train_rsepm.py \
--gpus 0 --root_dir ../VOC2012/ --dataset VOC --train_domain train_aug \
--backbone resnet50 --tag ResNet50@VOC@RS+EPM

# 2. MS COCO 2014
python3 train_rsepm.py \
--gpus 0 --root_dir ../COCO2014/ --dataset COCO --train_domain train \
--backbone resnet50 --tag ResNet50@COCO@RS+EPM
```

### 2. AffinityNet For Random Walk (multi-stage)
We apply a random walk for improving the quality of pseudo masks.
Additionally, we implemented a random walk refer to [this repository](https://github.com/jiwoon-ahn/irn). 

```bash
# Generate initial seeds produced from our single-stage method.
python3 infer_rsepm.py \
--gpus 2 --root_dir ../VOC2012/ --dataset VOC --domain train_aug \
--backbone resnet50 --tag ResNet50@VOC@RS+EPM@Official

# Convert initial seeds to affinity labels for training AffinityNet.
python3 convert_seed_to_affinity_labels.py \
--root_dir ../VOC2012/ --dataset VOC --domain train_aug \
--tag ResNet50@VOC@RS+EPM@Official

# Train AffinityNet for a random walk.
python3 train_aff.py \
--gpus 1 --root_dir ../VOC2012/ --dataset VOC --train_domain train_aug \
--backbone resnet50 --tag ResNet50@VOC@AffinityNet \
--pseudo_tag ResNet50@VOC@RS+EPM@Official
```

### 3. Final Segmentation Model (multi-stage)
Additionally, we implemented DeepLabv2 and DeepLabv3+ refer to [this repository](https://github.com/rulixiang/deeplab-pytorch).

```bash
# Apply RW for improving the quality of single-stage results.
python3 infer_rw.py \
--gpus 3 --root_dir ../VOC2012/ --dataset VOC --domain train_aug \
--backbone resnet50 --aff_tag ResNet50@VOC@AffinityNet \
--seg_tag ResNet50@VOC@RS+EPM@Official

# Convert initial seeds to pseudo masks for training final segmentation models.
python3 convert_seed_to_pseudo_masks.py \
--gpus 0 --root_dir ../VOC2012/ --dataset VOC --domain train_aug \
--folder random_walk --tag ResNet50@VOC@RS+EPM@Official

# Train DeepLabv3+ with pseudo masks.
python3 train_seg.py \
--gpus 1 --root_dir ../VOC2012/ --dataset VOC \
--train_domain train_aug --valid_domain validation \
--pseudo_tag ResNet50@VOC@RS+EPM@Official \
--backbone resnet101 --decoder deeplabv3+ \
--tag ResNet101@VOC@RS+EPM@MS
```

# Evaluation

Release our weights, predicted masks, and official results (anonymous link).

| Stage | Backbone     | Pretrained weight            | VOC val | VOC test |
|:-----:|:------------:|:----------------------------:|:-------:|:--------:|
| single-stage | ResNet-50 | [weight](https://drive.google.com/file/d/1i-RenkUjqsiy05-GcwRFzXXRKkV9MICA/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/LKO4IS.html) | [link](http://host.robots.ox.ac.uk:8080/anonymous/9MLLHH.html) |
| multi-stage (DeepLabv2) | ResNet-101 | [weight](https://drive.google.com/file/d/1-k1M_H2hvCUaGJlos1T3ivqJ6G5y5dD-/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/GO5RTP.html) [mask](https://drive.google.com/file/d/1iHJaEpJVa6G8l4Bd1rd8LKhGRNPX9rzB/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/0D2TEN.html) [mask](https://drive.google.com/file/d/1QPAOI971N8q-xCL2lVM9lHLXPBc-zalg/view?usp=sharing) |
| multi-stage (DeepLabv3+) | ResNet-101 | [weight](https://drive.google.com/file/d/1Z5zOC0mXIqVL4umUcfujoMuirFfiieup/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/SNDUAQ.html) [mask](https://drive.google.com/file/d/1TZXmJm7UEO1u3uqss-GF2LyClUFc1amY/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/HUHADT.html) [mask](https://drive.google.com/file/d/1PmuZ3wRirsvh09hCR1inpxhmkeoF0Id0/view?usp=sharing) |
| multi-stage (DeepLabv3+, COCO) | ResNet-101 | [weight](https://drive.google.com/file/d/1UwGM_0P-eR6O1T8ydRdxvBPHZiZ6eJHO/view?usp=share_link) | - | - |

Release pseudo labels on PASCAL VOC 2012 dataset after applying a random walk with our single-stage method. [RS+EPM.zip](https://drive.google.com/file/d/1MwqpQQrD7ANOI3x-AlCoAWFQLnnpcU2W/view?usp=sharing)

Below lines are testing commands to reproduce our method.

Before running the testing commands, download the official checkpoints ([ResNet50@VOC@RS+EPM@Official.zip](https://drive.google.com/file/d/1i-RenkUjqsiy05-GcwRFzXXRKkV9MICA/view?usp=sharing) and [ResNet101@VOC@RS+EPM@MS@Official.zip](https://drive.google.com/file/d/1Z5zOC0mXIqVL4umUcfujoMuirFfiieup/view?usp=sharing)) and extract each zip to have the following structure.
```
./experiments/models/ResNet50@VOC@RS+EPM@Official/last.pth
./experiments/models/ResNet101@VOC@RS+EPM@MS@Official/last.pth
```

### 1. Single-stage Results
```bash
# Generate initial seeds produced from our single-stage method.
python3 infer_rsepm.py \
--gpus 2 --root_dir ../VOC2012/ --dataset VOC --domain validation \
--backbone resnet50 --tag "ResNet50@VOC@RS+EPM@Official" --conf_th 0.34

# Convert initial seeds to predicted masks with CRF.
python3 convert_seed_to_pseudo_masks.py \
--gpus 0 --root_dir ../VOC2012/ --dataset VOC --domain validation \
--folder predictions --tag "ResNet50@VOC@RS+EPM@Official" 

# Calculate the mIoU.
# [validation] http://host.robots.ox.ac.uk:8080/anonymous/LKO4IS.html
# [test] http://host.robots.ox.ac.uk:8080/anonymous/9MLLHH.html
python3 evaluate.py \
--root_dir ../VOC2012/ --domain validation \
--folder pseudo-labels --tag "ResNet50@VOC@RS+EPM@Official"
```

### 2. Multi-stage Results
```bash
# Generate initial seeds produced from final segmentation model.
python3 infer_seg.py \
--gpus 3 --root_dir ../VOC2012/ --dataset VOC --domain validation \
--backbone resnet101 --decoder deeplabv3+ \
--tag "ResNet101@VOC@RS+EPM@MS@Official"

# Convert initial seeds to predicted masks with CRF.
python3 convert_seed_to_pseudo_masks.py \
--gpus 0 --root_dir ../VOC2012/ --dataset VOC --domain validation \
--folder predictions --tag "ResNet101@VOC@RS+EPM@MS@Official"

# Calculate the mIoU.
# [validation] http://host.robots.ox.ac.uk:8080/anonymous/SNDUAQ.html
# [test] http://host.robots.ox.ac.uk:8080/anonymous/HUHADT.html
python3 evaluate.py \
--root_dir ../VOC2012/ --domain validation \
--folder pseudo-labels --tag "ResNet101@VOC@RS+EPM@MS@Official"
```