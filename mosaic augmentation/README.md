# Mosaic Data Augmentation

Mosaic data augmentation combines 4 training images into one in certain ratios (instead of only two in CutMix). [Mosaic](https://www.youtube.com/watch?v=V6uj-eGmE7g) is the first new data augmentation technique introduced in [YOLOv4](https://arxiv.org/pdf/1905.04899.pdf). This allows for the model to learn how to identify objects at a smaller scale than normal. It also is useful in training to significantly reduce the need for a large mini-batch size.

Demo with bbox:

![image](https://github.com/bharatmahaur/ComparativeStudy/blob/main/mosaic%20augmentation/reg_full_1.jpg)

Dataset come from:

[BDD100K](https://doc.bdd100k.com/download.html)
