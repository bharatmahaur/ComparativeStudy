##
## /src/models/ssd/common/np_utils.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 07/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import numpy as np
import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


def np_jaccard_score_single(box, # (4,) <- (ymin, xmin, ymax, xmax)
							anchor_boxes): # (num_aboxes, 4) <- (ymin, xmin, ymax, xmax)
	"""Compute Jaccard score (Intersection over Union) of one bounding box to multiple anchor boxes.

	Arguments:
		box: Array of length 4 containing coordinates of reference box. Format: (y_min, x_min, y_max, x_max).
		anchor_boxes: Array of shape (num_aboxes, 4) containing coordinates of anchor boxes. Format: (y_min, x_min, y_max, x_max).

	Returns:
		Array of jaccard scores of shape (num_aboxes,).
	"""
	y_min = np.maximum(anchor_boxes[:, 0], box[:, 0]) # (num_aboxes,)
	x_min = np.maximum(anchor_boxes[:, 1], box[:, 1]) # (num_aboxes,)
	y_max = np.minimum(anchor_boxes[:, 2], box[:, 2]) # (num_aboxes,)
	x_max = np.minimum(anchor_boxes[:, 3], box[:, 3]) # (num_aboxes,)

	box_volume = (box[2] - box[0]) * (box[3] - box[1]) # ()
	abox_volumes = (anchor_boxes[:, 2] - anchor_boxes[:, 0]) * (anchor_boxes[:, 3] - anchor_boxes[:, 1]) # (num_aboxes,)

	intersection_volumes = np.maximum(y_max - y_min, 0.0) * np.maximum(x_max - x_min, 0.0) # (num_aboxes,)
	union_volumes = abox_volumes + box_volumes - intersection_volumes # (num_aboxes,)

	jaccard_score = intersection_volumes / union_volumes # (num_aboxes,)
	return jaccard_score # (num_aboxes,)


def np_jaccard_score(boxes, # (N, 4) <- (ymin, xmin, ymax, xmax)
					 anchor_boxes): # (num_aboxes, 4) <- (ymin, xmin, ymax, xmax)
	"""Compute Jaccard score (Intersection over Union) of multiple bounding boxes to multiple anchor boxes.

	Arguments:
		box: Array of shape (N, 4) containing coordinates of reference boxes. Format: (y_min, x_min, y_max, x_max).
		anchor_boxes: Array of shape (num_aboxes, 4) containing coordinates of anchor boxes. Format: (y_min, x_min, y_max, x_max).

	Returns:
		Array of jaccard scores of shape (N, num_aboxes).
	"""
	boxes = np.expand_dims(boxes, axis=-1) # (N, 4, 1) <- (ymin, xmin, ymax, xmax)

	y_min = np.maximum(anchor_boxes[:, 0], boxes[:, 0]) # (N, num_aboxes)
	x_min = np.maximum(anchor_boxes[:, 1], boxes[:, 1]) # (N, num_aboxes)
	y_max = np.minimum(anchor_boxes[:, 2], boxes[:, 2]) # (N, num_aboxes)
	x_max = np.minimum(anchor_boxes[:, 3], boxes[:, 3]) # (N, num_aboxes)

	box_volumes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # (N, 1)
	abox_volumes = (anchor_boxes[:, 2] - anchor_boxes[:, 0]) * (anchor_boxes[:, 3] - anchor_boxes[:, 1]) # (num_aboxes,)

	intersection_volumes = np.maximum(y_max - y_min, 0.0) * np.maximum(x_max - x_min, 0.0) # (N, num_aboxes)
	union_volumes = abox_volumes + box_volumes - intersection_volumes # (N, num_aboxes)

	jaccard_score = intersection_volumes / union_volumes # (N, num_aboxes)
	return jaccard_score # (N, num_aboxes)


def np_ssd_default_anchor_boxes(image_shape,
								feature_shapes,
								anchor_ratios,
								anchor_steps,
								anchor_scales,
								anchor_extra_scale,
								anchor_offset):
	"""Compute default anchor boxes given some hyperparameters.

	Arguments:
		image_shape: Tuple defining image shape of network. Format: (height, width).
		feature_shapes: List of tuples defining shapes of feature layers. Format: (height, width).
		anchor_ratios: Aspect ratios of feature layers.
		anchor_steps: Steps for defining the locations of the feature layers.
		anchor_scales: Scales of default anchor boxes of feature layers.
		anchor_extra_scale: Additional scale of last feature layer.
		anchor_offset: Offset of anchor box locations.

	Returns:
		Default anchor boxes of all feature layers of shape (num_aboxes, 4). Format: (cy, cx, height, width).
	"""
	anchor_boxes = []

	# iterate feature layers
	for ii, feature_shape in enumerate(feature_shapes):
		# compute mesh grids
		y, x = np.mgrid[0:feature_shape[0], 0:feature_shape[1]]
		y = (y + anchor_offset) * anchor_steps[ii] / image_shape[0]
		x = (x + anchor_offset) * anchor_steps[ii] / image_shape[1]

		y = np.expand_dims(y, axis=-1)
		x = np.expand_dims(x, axis=-1)

		# prepare scales and ratios
		scale = anchor_scales[ii]
		ratios = [1.0] + anchor_ratios[ii]
		ratios = np.sqrt(ratios)

		height = []
		width = []

		# create boxes of different ratios
		for ratio in ratios:
			height.append(scale / ratio)
			width.append(scale * ratio)

		# create box with additional scale from next feature layer
		if ii < len(anchor_scales) - 1:
			sprime = np.sqrt(scale * anchor_scales[ii + 1])
		else:
			sprime = np.sqrt(scale * anchor_extra_scale)
		height.append(sprime)
		width.append(sprime)

		# append heights and weights to result
		height = np.array(height)
		width = np.array(width)
		anchor_boxes.append((y, x, height, width))

	# merge anchor boxes to a single array
	merged_anchors = []
	for ii, single_anchor_boxes in enumerate(anchor_boxes):
		num_sizes = single_anchor_boxes[2].shape[0]
		num_boxes = single_anchor_boxes[0].shape[0] * single_anchor_boxes[1].shape[0] * num_sizes

		y_flat = np.repeat(single_anchor_boxes[0], num_sizes)
		x_flat = np.repeat(single_anchor_boxes[1], num_sizes)
		height_flat = np.tile(single_anchor_boxes[2], num_boxes // num_sizes)
		width_flat = np.tile(single_anchor_boxes[3], num_boxes // num_sizes)

		merged_anchors.append(
			np.stack([y_flat, x_flat, height_flat, width_flat], axis=-1)
		)

	return np.vstack(merged_anchors) # (num_aboxes, 4) <- (cy, cx, h, w)
