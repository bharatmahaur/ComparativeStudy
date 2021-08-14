##
## /src/models/ssd/sdd_vgg_512.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 16/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import math
import numpy as np
import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from models.custom_layers import atrous_convolution2d, convolution2d, l2_normalization, max_pooling2d, smooth_l1_loss
from models.ssd.common import np_ssd_default_anchor_boxes, tf_ssd_decode_boxes, tf_ssd_select_boxes, tf_ssd_suppress_overlaps
from models.ssd.ssd_vgg_base import SSDVGGBase
from utils.common.files import get_full_path


class SSDVGG512(SSDVGGBase):

	default_hyperparams = {
		"image_shape": (512, 512),
		"tf_image_shape": [512, 512, 3],
		"num_classes": 20 + 1,
		"feature_layers": ["block_4", "block_7", "block_8", "block_9", "block_10", "block_11", "block_12"],
		"feature_shapes": [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
		"feature_normalizations": [20, None, None, None, None, None, None],
		"anchor_scales": [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9],
		"anchor_extra_scale": 1.06,
		"anchor_ratios": [
			[2.0, 0.5],
			[2.0, 3.0, 0.5, 1.0 / 3.0],
			[2.0, 3.0, 0.5, 1.0 / 3.0],
			[2.0, 3.0, 0.5, 1.0 / 3.0],
			[2.0, 3.0, 0.5, 1.0 / 3.0],
			[2.0, 0.5],
			[2.0, 0.5]
		],
		"anchor_steps": [8, 16, 32, 64, 128, 265, 512],
		"anchor_offset": 0.5,
		"num_anchors": 24564,
		"prior_scaling": [0.1, 0.1, 0.2, 0.2],
		"matching_threshold": 0.5,
		"keep_probability": 0.5
	}
	network_name = "ssd_vgg_512"
