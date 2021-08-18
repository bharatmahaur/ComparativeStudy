# Train deep learning-based models (R-FCN, Mask R-CNN, SSD, RetinaNet and YOLOv4) for road object detection.

This repository is an implementation of a review paper on Python 3 (TensorFlow and Caffe).

## Experimental Setup

Name | Supported Versions
--- | --- |
Ubuntu |18.04.5
Python | 3.6
CUDA | 10.1
Cudnn | 7.6.4
OpenCV | 4.5.0
Tensorflow | 2.1+
Caffe | 1.0

To install requirements virtualenv and virtualenvwrapper should be available on the target machine.

**Virtual Environment Creation:**
```
# Clone repo
git clone https://github.com/bharatmahaur/ComparativeStudy.git

# Create python virtual env
mkvirtualenv ComparativeStudy

# Add library path to virtual env
add2virtualenv ComparativeStudy

```

## Datasets
Download the Berkeley Deep Drive (BDD) Object Detection Dataset [here](https://bdd-data.berkeley.edu/). The BDD
dataset should have the following structure:
<br>
 
     └── BDD_DATASET_ROOT
         ├── info
         |   └── 100k
         |       ├── train
         |       └── val
         ├── labels
         └── images
                ├── 10K
                └── 100K
                    ├── test
                    ├── train
                    └── val
<br> 

## Mosiac Augmentation
Go to the mosiac folder run :

## Training and Evaluation
To train the model(s) in the paper, go to the folder:

## Trained Weights

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

## Citation
If you use this code, please cite our paper:
```
@misc{bharat2021review,
title={Paper under review}, 
author={Bharat Mahaur, et al.},
year={2021},
```

## License
This code is released under the [Apache 2.0 License](LICENSE.md).
