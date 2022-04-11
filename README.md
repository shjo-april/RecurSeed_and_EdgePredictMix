# RecurSeed and CertainMix
This repository is the official implementation of "RecurSeed and CertainMix for Weakly Supervised Semantic Segmentation". Please feel free to reach out for any questions or discussions.

### Abstract
Although weakly supervised semantic segmentation using only image-level labels (WSSS-IL) is potentially useful, its low performance and implementation complexity still limit its use. The main causes are (a) non-detection and (b) false-detection phenomena: (a) The class activation maps refined from existing WSSS-IL methods still only represent partial regions for large-scale objects, (b) and for small-scale objects, over-activations cause them to deviate from the object edges. We propose RecurSeed which alternately reduces non- and false-detections through recursive iterations, thereby implicitly finding an optimal junction minimizing both errors. To maximize the effectiveness of RecurSeed, we also propose a novel data augmentation (DA)  approach called CertainMix, which virtually creates object masks and expresses their edges in more detail by combining the segmentation results, thereby achieving a DA more spatially matching the ground-truth masks. We achieved new state-of-the-art performances on both the PASCAL VOC 2012 and MS COCO 2014 benchmarks (VOC \emph{val}: $72.4\%$, COCO \emph{val}: $45.0\%$).

![Overview](./resources/Overview.jpg)


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
> Download PASCAL VOC 2012 dataset from our [[Google Drive](https://drive.google.com/file/d/1dkwHjd-r4Xe4ap0PWNMn0GRnekIrEKyQ/view?usp=sharing)].

> ##### 2. MS COCO 2014
> Download MS COCO 2014 dataset from our [[Google Drive](https://drive.google.com/file/d/1Nn2zsJg3L52xYo40s3nx_EeUNAa4RULf/view)].

Create a directory "../VOC2012/" for storing the dataset and appropriately place each dataset to have the following directory structure.
```
    ../                               # parent directory
    ├── ./                            # current (project) directory
    │   ├── core/                     # (dir.) implementation of RecurSeed and CertainMix
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
We prepared [a jupyter notebook](https://github.com/OFRIN/RecurSeed_and_CertainMix/blob/master/demo.ipynb) for visualization.

# Training
The whole code and commands are under internal review and will release soon.

# Evaluation
Release our weights, official results, and final masks through our methods.

| Stage | Architecture | Pretrained weight            | VOC val | VOC test |
|:-----:|:------------:|:----------------------------:|:-------:|:--------:|
| single-stage | DeepLabv3+ | [weight](https://drive.google.com/file/d/1KtIGxmqf3FeIETs-rc3hE9pUygcKpeNI/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/M6YRQV.html) [mask](https://drive.google.com/file/d/1WRDe000_rRHDdWC4183cFQZztnDEwNBO/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/Z99QQ9.html) [mask](https://drive.google.com/file/d/1kAYw3fM18KDC_CpCrx3C1iwF3DNw55Zk/view?usp=sharing) |
| multi-stage | DeepLabv2 | [weight](https://drive.google.com/file/d/1KtIGxmqf3FeIETs-rc3hE9pUygcKpeNI/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/GETYD6.html) [mask](https://drive.google.com/file/d/1ldTWo2VtFH2jLG5Zip7ZFnZZjJHNCOho/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/ZEOASL.html) [mask](https://drive.google.com/file/d/11h9kHfTpTY97Kl0DJ0U6SINgH3bDCKHt/view?usp=sharing) |
| multi-stage | DeepLabv3+ | [weight](https://drive.google.com/file/d/1KtIGxmqf3FeIETs-rc3hE9pUygcKpeNI/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/2XIMJS.html) [mask](https://drive.google.com/file/d/1HZyuZpfREhmALwy9T1dFZJUq1KANc1u3/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/4QJDCS.html) [mask](https://drive.google.com/file/d/1I_mftMmXdC8ZFFBOmAaSju9j9vSvlDk8/view?usp=sharing) |

# Citation
```
```