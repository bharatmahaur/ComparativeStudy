# Road object detection: a comparative study of deep learning-based algorithms.

This repository is an implementation of a review article on Python 3 (TensorFlow and Caffe). We train deep learning-based models (R-FCN, Mask R-CNN, SSD, RetinaNet and YOLOv4) for road object detection on BDD100K dataset.<br />

Please find the paper here: [https://link.springer.com/article/10.1007/s11042-022-12447-5](https://link.springer.com/article/10.1007/s11042-022-12447-5).

## Requirements

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

## Structure

```
├─ Mosiac                           <- Convert to mosaic images and xml files for training models
├─ R-FCN                            <- R-FCN model training and evaluation on metrices
├─ Mask R-CNN                       <- Mask R-CNN model training and evaluation on metrices
├─ SSD                              <- SSD model training and evaluation on metrices
├─ RetinaNet                        <- RetinaNet model training and evaluation on metrices
├─ YOLOv4                           <- YOLOv4 model training and evaluation on metrices
├─ LICENSE.md
└─ README.md
```

## Dataset
Download the Berkeley Deep Drive (BDD100K) Object Detection Dataset [here](https://bdd-data.berkeley.edu/). The BDD
dataset has the following structure:
<br>
 
     └── BDD100K_DATASET_ROOT
         ├── info
         |   └── 100k
         |       ├── train
         |       └── val
         |       └── test         
         ├── labels
         └── images
                └── 100K
                    ├── test
                    ├── train
                    └── val
<br> 

## Mosiac Augmentation
1. Go to the mosiac augmentation folder, run [mosaic_data.ipynb](https://github.com/bharatmahaur/ComparativeStudy/blob/main/mosaic%20augmentation/mosaic_data.ipynb)
2. Use the BDD100K downloaded files or try [sample](https://github.com/bharatmahaur/ComparativeStudy/tree/main/mosaic%20augmentation/sample) folder 
3. Generate mosiac xml and output images like:

<img src="https://github.com/bharatmahaur/ComparativeStudy/blob/main/mosaic%20augmentation/reg_full_1.jpg" width="auto" height="250">

## Training and Evaluation
To train and evaluate the model(s) use the mosiac output files or use your own custom dataset and go to the individual model folder for furthur instructions.

## Trained Models

We trained these models on BDD100K, use the weights for predictions:
1. R-FCN:  [https://drive.google.com/file/d/11lqFSrRVDZViJCaaKFNivVTi7gPYxLlx/view](https://drive.google.com/file/d/11lqFSrRVDZViJCaaKFNivVTi7gPYxLlx/view?usp=sharing)
2. Mask R-CNN:  [https://drive.google.com/file/d/1JRm5chovHuNm4pU8czBHnEAj4SXXP2Mz/view](https://drive.google.com/file/d/1JRm5chovHuNm4pU8czBHnEAj4SXXP2Mz/view?usp=sharing)
3. SSD:  [https://drive.google.com/file/d/1SCb_5z1vhTIn3pp-VeA-KmLjejbfhlfI/view](https://drive.google.com/file/d/1SCb_5z1vhTIn3pp-VeA-KmLjejbfhlfI/view?usp=sharing)
4. RetinaNet:  [https://drive.google.com/file/d/13P26Bb-9IiyEMx18JNlnvjrSZH-CP8x1/view](https://drive.google.com/file/d/13P26Bb-9IiyEMx18JNlnvjrSZH-CP8x1/view?usp=sharing)
5. YOLOv4:  [https://drive.google.com/file/d/1k-6Y4nGnelSOO7fg6gF6R58Zbu12W4TG/view](https://drive.google.com/file/d/1k-6Y4nGnelSOO7fg6gF6R58Zbu12W4TG/view?usp=sharing)

## Citation
If you use this code, please cite our paper:
```
@article{mahaur2022road,
 title={Road object detection: a comparative study of deep learning-based algorithms}, 
 author={Mahaur, Bharat and others},
 journal={Multimedia Tools and Applications},
 year={2022},
 publisher={Springer}
}
```

## Contact
Please contact [bharatmahaur@gmail.com](mailto:bharatmahaur@gmail.com) for any further queries.

## License
This code is released under the [Apache 2.0 License](LICENSE.md).
