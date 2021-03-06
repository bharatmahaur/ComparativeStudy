# YOLOv4 with CSPDarkNet-53

- Paper : https://arxiv.org/abs/2004.10934
- Source code: https://github.com/AlexeyAB/darknet

## Overview
+ Backbone：CSPDarkNet53
+ Neck：SPP，PAN
+ Head：YOLOv3
+ Tricks (Backbone): CutMix、Mosaic、DropBlock、Label Smoothing
+ Modified(Backbone) : Mish、CSP、MiWRC
+ Tricks (Detection) : CIoU、CmBN、SAT、Eliminate grid sensitivity
+ Modified(Detection): Mish、SPP、SAM、PAN、DIoU-NMS

## Usage
### Train on BDD100K to PASCAL VOC 2012
```
|——data
    |——dataset 
        |——VOCdevkit
            |——VOC2012
                |——Annotations
                |——ImageSets
                |——JPEGImages
                |——SegmentationClass
                |——SegmentationObject
```
1. Unzip the file and place it in the 'dataset' folder, make sure the directory is like this : 
2. Run ./data/write_voc_to_txt.py to generate voc2012.txt, which operation is essential. 
3. Run train.py

## Project Schedule
### Data augmentation
- [ ] Mosaic
- [ ] Cutmix
- [ ] Self-adversarial-training (SAT)
### Model
- [x] Cross-stage partial Net (CSP-DarkNet53)
- [x] Mish-activation
- [x] DropBlock regularization
- [x] SPP-block
- [ ] SAM-block
- [x] PAN block
- [ ] Cross mini-Batch Normalization (CmBN)
### Otimization
- [ ] Multi-input weighted residual connections (MiWRC)
- [ ] Eliminate grid sensitivity
- [x] Cosine annealing scheduler
- [ ] kmeans
- [x] DIoU-NMS
### Loss
- [x] Class label smoothing
- [x] CIoU loss
- [x] Focal loss

