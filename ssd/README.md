# SSD with VGG-16

- Paper : https://arxiv.org/abs/1512.02325
- Source code: https://github.com/weiliu89/caffe

## Structure

```
.
├─ data/
│  ├─ raw/                   <- downloaded archives
│  │  ├─ voc2007/
│  │  └─ voc2012/
│  ├─ interim/               <- extracted archives
│  └─ processed/             <- converted datasets
├─ docs/                     <- project documentation
│  ├─ poster/
│  └─ report/
├─ models/                   <- pre-trained weights and frozen models
│  └─ vgg_16_imagenet/       <- pre-trained VGG 16 weights
├─ src/
│  ├─ data/                  <- data input pipeline
│  │  └─ preprocessors/      <- data pre-processors
│  ├─ datasets/              <- extract and convert datasets
│  │  └─ common/
│  ├─ models/                <- model implementation
│  │  ├─ custom_layers/
│  │  └─ ssd/
│  ├─ utils/                 <- utility functions and classes
│  │  └─ common/
│  ├─ eval.py                <- evaluate a model using mean average precision
│  ├─ freeze.py              <- freeze a trained model for faster inference
│  ├─ infer.py               <- start inference with a single image
│  └─ train.py               <- train a new model
├─ training/                 <- run configurations and saved checkpoints
│  └─ run_*/                    created by src/train.py
└─ README.md
```

## Getting Started

To get started, download the pre-trained [VGG 16 weights](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and extract the file `vgg_16.ckpt` to `tf-ssd-vgg/models/vgg_16_imagenet`.

To train the neural network, the Pascal VOC 2007 and 2012 datasets were used. Download the following archives and move them to `tf-ssd-vgg/data/raw/voc2007` and `tf-ssd-vgg/data/raw/voc2012`.
- [VOC 2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
- [VOC 2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
- [VOC 2012 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

