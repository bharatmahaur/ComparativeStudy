##
## /src/models/ssd/ssd_vgg_300.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 08/07/2018.
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


class SSDVGG300(SSDVGGBase):

	default_hyperparams = {
		"image_shape": (300, 300),
		"tf_image_shape": [300, 300, 3],
		"num_classes": 20 + 1,
		"feature_layers": ["block_4", "block_7", "block_8", "block_9", "block_10", "block_11"],
		"feature_shapes": [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
		"feature_normalizations": [20, None, None, None, None, None],
		"anchor_scales": [0.07, 0.15, 0.33, 0.51, 0.69, 0.87],
		"anchor_extra_scale": 1.05,
		"anchor_ratios": [
			[2.0, 0.5],
			[2.0, 3.0, 0.5, 1.0 / 3.0],
			[2.0, 3.0, 0.5, 1.0 / 3.0],
			[2.0, 3.0, 0.5, 1.0 / 3.0],
			[2.0, 0.5],
			[2.0, 0.5]
		],
		"anchor_steps": [8, 16, 32, 64, 100, 300],
		"anchor_offset": 0.5,
		"num_anchors": 8732,
		"prior_scaling": [0.1, 0.1, 0.2, 0.2],
		"matching_threshold": 0.5,
		"keep_probability": 0.5
	}
	network_name = "ssd_vgg_300"
