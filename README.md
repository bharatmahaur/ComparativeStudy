# Train deep learning-based models (R-FCN, Mask R-CNN, SSD, RetinaNet and YOLOv4) for road object detection.

This repository is an implementation of a review paper on Python 3 (TensorFlow and Caffe).

## Experimental Requirements

Name | Supported Versions
--- | --- |
Ubuntu |18.04.5
Python | 3.6
CUDA | 10.1
Cudnn | 7.6.4
OpenCV | 4.5.0

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
1. Go to the mosiac augmentation folder, run [mosaic_data.ipynb](https://github.com/bharatmahaur/ComparativeStudy/blob/main/mosaic%20augmentation/mosaic_data.ipynb)
2. Use the dataset files or [sample](https://github.com/bharatmahaur/ComparativeStudy/tree/main/mosaic%20augmentation/sample) folder 
3. Generate xml and output images like:

<img src="https://github.com/bharatmahaur/ComparativeStudy/blob/main/mosaic%20augmentation/reg_full_1.jpg" width="auto" height="250">

## Training and Evaluation
To train the model(s) use the mosiac output files and go to the individual folder for furthur instructions.

## Trained Weights

Avaliable soon:
1. R-FCN
2. Mask R-CNN
3. SSD
4. RetinaNet
5. Yolov4

## Citation
If you use this code, please cite our paper:
```
@article{,
title={}, 
author={},
year={},
}
```

## License
This code is released under the [Apache 2.0 License](LICENSE.md).
