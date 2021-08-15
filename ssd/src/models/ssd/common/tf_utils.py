##
## /src/models/ssd/common/tf_utils.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 07/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


def tf_jaccard_score_single(box, # (4,) <- (ymin, xmin, ymax, xmax)
							anchor_boxes): # (num_aboxes, 4) <- (ymin, xmin, ymax, xmax)
	"""Compute Jaccard score (Intersection over Union) of one bounding box to multiple anchor boxes.

	Arguments:
		box: Tensor of length 4 containing coordinates of reference box. Format: (y_min, x_min, y_max, x_max).
		anchor_boxes: Tensor of shape (num_aboxes, 4) containing coordinates of anchor boxes. Format: (y_min, x_min, y_max, x_max).

	Returns:
		Tensor of jaccard scores of shape (num_aboxes,).
	"""
	y_min = tf.maximum(anchor_boxes[:, 0], box[0]) # (num_aboxes,)
	x_min = tf.maximum(anchor_boxes[:, 1], box[1]) # (num_aboxes,)
	y_max = tf.minimum(anchor_boxes[:, 2], box[2]) # (num_aboxes,)
	x_max = tf.minimum(anchor_boxes[:, 3], box[3]) # (num_aboxes,)

	box_volumes = (box[2] - box[0]) * (box[3] - box[1]) # ()
	abox_volumes = (anchor_boxes[:, 2] - anchor_boxes[:, 0]) * (anchor_boxes[:, 3] - anchor_boxes[:, 1]) # (num_aboxes,)

	intersection_volumes = tf.maximum(y_max - y_min, 0.0) * tf.maximum(x_max - x_min, 0.0) # (num_aboxes,)
	union_volumes = abox_volumes + box_volumes - intersection_volumes # (num_aboxes,)

	jaccard_score = intersection_volumes / union_volumes # (num_aboxes,)
	return jaccard_score # (num_aboxes,)


def tf_jaccard_score(boxes, # (N, 4) <- (ymin, xmin, ymax, xmax)
					 anchor_boxes): # (num_aboxes, 4) <- (ymin, xmin, ymax, xmax)
	"""Compute Jaccard score (Intersection over Union) of multiple bounding boxes to multiple anchor boxes.

	Arguments:
		box: Tensor of shape (N, 4) containing coordinates of reference boxes. Format: (y_min, x_min, y_max, x_max).
		anchor_boxes: Tensor of shape (num_aboxes, 4) containing coordinates of anchor boxes. Format: (y_min, x_min, y_max, x_max).

	Returns:
		Tensor of jaccard scores of shape (N, num_aboxes).
	"""
	boxes = tf.expand_dims(boxes, axis=-1) # (N, 4, 1) <- (ymin, xmin, ymax, xmax)

	y_min = tf.maximum(anchor_boxes[:, 0], boxes[:, 0]) # (N, num_aboxes)
	x_min = tf.maximum(anchor_boxes[:, 1], boxes[:, 1]) # (N, num_aboxes)
	y_max = tf.minimum(anchor_boxes[:, 2], boxes[:, 2]) # (N, num_aboxes)
	x_max = tf.minimum(anchor_boxes[:, 3], boxes[:, 3]) # (N, num_aboxes)

	box_volumes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # (N, 1)
	abox_volumes = (anchor_boxes[:, 2] - anchor_boxes[:, 0]) * (anchor_boxes[:, 3] - anchor_boxes[:, 1]) # (num_aboxes,)

	intersection_volumes = tf.maximum(y_max - y_min, 0.0) * tf.maximum(x_max - x_min, 0.0) # (N, num_aboxes)
	union_volumes = abox_volumes + box_volumes - intersection_volumes # (N, num_aboxes)

	jaccard_score = intersection_volumes / union_volumes # (N, num_aboxes)
	return jaccard_score # (N, num_aboxes)


def tf_ssd_encode_boxes(labels, # (N,)
						boxes, # (N, 4) <- (ymin, xmin, ymax, xmax)
						anchor_boxes, # (num_aboxes, 4) <- (cy, cx, h, w)
						prior_scaling,
						matching_threshold = 0.5):
	"""Match and encode groundtruth boxes using default anchor boxes for the SSD network.

	This function does not work when N = 0, i.e. no groundtruth boxes are given. That happened when cropping the image while pre-processing
	with bad hyperparameters.

	Arguments:
		labels: Tensor of length N containing labels of groundtruth boxes.
		boxes: Tensor of shape (N, 4) containing groundtruth bounding boxes.
		anchor_boxes: Array of shape (num_aboxes, 4) containing default anchor boxes.
		prior_scaling: List containing scaling of SSD boxes.
		matching_threshold: IoU threshold of matching. Defaults to 0.5.

	Returns:
		feauture_labels: Tensor of shape (num_aboxes,) containing encoded labels.
		feature_scores: Tensor of shape (num_aboxes,) containing encoded scores.
		feature_boxes: Tensor of shape (num_aboxes, 4) containing encoded boxes.
	"""
	num_boxes = tf.cast(tf.shape(boxes)[0], tf.int64) # ()
	num_aboxes = anchor_boxes.shape[0] # ()

	# compute coordinates of anchor boxes
	aboxes_ymin = tf.cast(anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2.0, tf.float32) # (num_aboxes,)
	aboxes_xmin = tf.cast(anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2.0, tf.float32) # (num_aboxes,)
	aboxes_ymax = tf.cast(anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2.0, tf.float32) # (num_aboxes,)
	aboxes_xmax = tf.cast(anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2.0, tf.float32) # (num_aboxes,)

	# compute Jaccard score of boxes to anchor boxes
	jaccard = tf_jaccard_score(
		boxes,
		tf.stack([aboxes_ymin, aboxes_xmin, aboxes_ymax, aboxes_xmax], axis=-1)
	) # (N, num_aboxes)

	# best and good abox for each box
	best_abox_per_box = tf.argmax(jaccard, axis=1) # (N,)
	best_abox_per_box = tf.expand_dims(best_abox_per_box, axis=-1) # (N, 1)
	best_abox_per_box_mask = tf.equal(tf.expand_dims(tf.range(num_aboxes, dtype=tf.int64), axis=0), best_abox_per_box) # (N, num_aboxes)
	good_abox_per_box_mask = tf.greater(jaccard, matching_threshold) # (N, num_aboxes)

	mask = tf.logical_or(best_abox_per_box_mask, good_abox_per_box_mask) # (N, num_aboxes)

	# best box for each abox
	jaccard_masked = tf.where(mask, jaccard, tf.zeros_like(mask, dtype=tf.float32)) # (N, num_aboxes)
	best_jaccard_per_abox = tf.reduce_max(jaccard_masked, axis=0) # (num_aboxes,)
	best_box_per_abox = tf.argmax(jaccard_masked, axis=0) # (num_aboxes,)
	best_box_per_abox = tf.expand_dims(best_box_per_abox, axis=0) # (1, num_aboxes)
	best_box_per_abox_mask = tf.equal(tf.expand_dims(tf.range(num_boxes, dtype=tf.int64), axis=-1), best_box_per_abox) # (N, num_aboxes)

	mask = tf.logical_and(mask, best_box_per_abox_mask) # (N, num_aboxes)
	imask = tf.cast(mask, tf.int64) # (N, num_aboxes)
	fmask = tf.cast(mask, tf.float32) # (N, num_aboxes)

	# update labels and boxes using the mask
	update_mask = tf.logical_not(tf.equal(tf.reduce_sum(imask, axis=0), 0)) # (num_aboxes,)

	labels = tf.expand_dims(labels, axis=-1) # (N, 1)
	labels = tf.multiply(labels, imask) # (N, num_aboxes)
	labels = tf.reduce_max(labels, axis=0) # (num_aboxes,)

	boxes = tf.expand_dims(boxes, axis=1) # (N, 1, 4)
	boxes = tf.multiply(boxes, tf.expand_dims(fmask, axis=-1)) # (N, num_aboxes, 4)
	boxes = tf.reduce_max(boxes, axis=0) # (num_aboxes, 4)
	boxes = tf.transpose(boxes, [1, 0]) # (4, num_aboxes)

	# build final labels, scores and box coordinates
	izeros = tf.zeros((num_aboxes,), dtype=tf.int64) # (num_aboxes,)
	fzeros = tf.zeros((num_aboxes,), dtype=tf.float32) # (num_aboxes,)
	fones = tf.ones((num_aboxes,), dtype=tf.float32) # (num_aboxes,)

	feature_labels = tf.where(update_mask, labels, izeros) # (num_aboxes,)
	feature_scores = tf.where(update_mask, best_jaccard_per_abox, fzeros) # (num_aboxes,)
	feature_y_min = tf.where(update_mask, boxes[0], fzeros) # (num_aboxes,)
	feature_x_min = tf.where(update_mask, boxes[1], fzeros) # (num_aboxes,)
	feature_y_max = tf.where(update_mask, boxes[2], fones) # (num_aboxes,)
	feature_x_max = tf.where(update_mask, boxes[3], fones) # (num_aboxes,)

	# transform to center / size
	feature_cy = (feature_y_max + feature_y_min) / 2.0 # (num_aboxes,)
	feature_cx = (feature_x_max + feature_x_min) / 2.0 # (num_aboxes,)
	feature_h = feature_y_max - feature_y_min # (num_aboxes,)
	feature_w = feature_x_max - feature_x_min # (num_aboxes,)

	# encode features
	feature_cy = (feature_cy - anchor_boxes[:, 0]) / anchor_boxes[:, 2] / prior_scaling[0] # (num_aboxes,)
	feature_cx = (feature_cx - anchor_boxes[:, 1]) / anchor_boxes[:, 3] / prior_scaling[1] # (num_aboxes,)
	feature_h = tf.log(feature_h / anchor_boxes[:, 2]) / prior_scaling[2] # (num_aboxes,)
	feature_w = tf.log(feature_w / anchor_boxes[:, 3]) / prior_scaling[3] # (num_aboxes,)

	# reorder for ssd
	feature_boxes = tf.stack(
		[feature_cx, feature_cy, feature_w, feature_h],
		axis = -1
	) # (num_aboxes, 4)

	return (
		feature_labels, # (num_aboxes,)
		feature_scores, # (num_aboxes,)
		feature_boxes # (num_aboxes, 4)
	)


def tf_ssd_encode_boxes_loop(labels, # (N,)
							 boxes, # (N, 4) <- (ymin, xmin, ymax, xmax)
							 anchor_boxes, # (num_aboxes, 4) <- (cy, cx, h, w)
							 prior_scaling,
							 matching_threshold = 0.5):
	"""Match and encode groundtruth boxes using default anchor boxes for the SSD network.

	This function is safe to use instead of `tf_ssd_encode_boxes`, even when no groundtruth boxes are given.

	Arguments:
		labels: Tensor of length N containing labels of groundtruth boxes.
		boxes: Tensor of shape (N, 4) containing groundtruth bounding boxes.
		anchor_boxes: Array of shape (num_aboxes, 4) containing default anchor boxes.
		prior_scaling: List containing scaling of SSD boxes.
		matching_threshold: IoU threshold of matching. Defaults to 0.5.

	Returns:
		feauture_labels: Tensor of shape (num_aboxes,) containing encoded labels.
		feature_scores: Tensor of shape (num_aboxes,) containing encoded scores.
		feature_boxes: Tensor of shape (num_aboxes, 4) containing encoded boxes.
	"""
	num_aboxes = anchor_boxes.shape[0] # ()

	# compute coordinates of anchor boxes
	aboxes_ymin = tf.cast(anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2.0, tf.float32) # (num_aboxes,)
	aboxes_xmin = tf.cast(anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2.0, tf.float32) # (num_aboxes,)
	aboxes_ymax = tf.cast(anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2.0, tf.float32) # (num_aboxes,)
	aboxes_xmax = tf.cast(anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2.0, tf.float32) # (num_aboxes,)
	aboxes = tf.stack([aboxes_ymin, aboxes_xmin, aboxes_ymax, aboxes_xmax], axis=-1)

	# initialize output tensors
	feature_labels = tf.zeros([num_aboxes], dtype=tf.int64) # (num_aboxes,)
	feature_scores = tf.zeros([num_aboxes], dtype=tf.float32) # (num_aboxes,)
	feature_y_min = tf.zeros([num_aboxes], dtype=tf.float32) # (num_aboxes,)
	feature_x_min = tf.zeros([num_aboxes], dtype=tf.float32) # (num_aboxes,)
	feature_y_max = tf.ones([num_aboxes], dtype=tf.float32) # (num_aboxes,)
	feature_x_max = tf.ones([num_aboxes], dtype=tf.float32) # (num_aboxes,)

	# define condition of the TF while loop
	def condition(ii,
				  feature_labels,
				  feature_scores,
				  feature_y_min,
				  feature_x_min,
				  feature_y_max,
				  feature_x_max):
		comparison = tf.less(ii, tf.shape(labels))
		return tf.gather(comparison, 0)

	# define body of the TF while loop
	def body(ii,
			 feature_labels,
			 feature_scores,
			 feature_y_min,
			 feature_x_min,
			 feature_y_max,
			 feature_x_max):
		label = tf.cast(tf.gather(labels, ii), tf.int64) # ()
		box = tf.gather(boxes, ii) # (4,)

		# compute Jaccard score of box to anchor boxes
		jaccard = tf_jaccard_score_single(box, aboxes) # (num_aboxes,)

		# best and good abox for each box
		best_abox_per_box = tf.argmax(jaccard, axis=0) # ()
		best_abox_per_box_mask = tf.equal(tf.range(num_aboxes, dtype=tf.int64), best_abox_per_box) # (num_aboxes,)
		good_abox_per_box_mask = tf.greater(jaccard, matching_threshold) # (num_aboxes,)

		mask = tf.logical_or(best_abox_per_box_mask, good_abox_per_box_mask) # (num_aboxes,)
		mask = tf.logical_and(mask, tf.greater(jaccard, feature_scores)) # (num_aboxes,)
		imask = tf.cast(mask, tf.int64) # (num_aboxes,)
		fmask = tf.cast(mask, tf.float32) # (num_aboxes,)

		# update labels and boxes using the mask
		feature_labels = imask * label + (1 - imask) * feature_labels # (num_aboxes,)
		feature_scores = tf.where(mask, jaccard, feature_scores) # (num_aboxes,)

		feature_y_min = fmask * tf.gather(box, 0) + (1.0 - fmask) * feature_y_min # (num_aboxes,)
		feature_x_min = fmask * tf.gather(box, 1) + (1.0 - fmask) * feature_x_min # (num_aboxes,)
		feature_y_max = fmask * tf.gather(box, 2) + (1.0 - fmask) * feature_y_max # (num_aboxes,)
		feature_x_max = fmask * tf.gather(box, 3) + (1.0 - fmask) * feature_x_max # (num_aboxes,)

		return [
			tf.add(ii, 1),
			feature_labels,
			feature_scores,
			feature_y_min,
			feature_x_min,
			feature_y_max,
			feature_x_max
		]

	ii = tf.constant(0, dtype=tf.int32)
	result = tf.while_loop(
		condition,
		body,
		[
			ii,
			feature_labels,
			feature_scores,
			feature_y_min,
			feature_x_min,
			feature_y_max,
			feature_x_max
		]
	)

	feature_labels = result[1]
	feature_scores = result[2]
	feature_y_min = result[3]
	feature_x_min = result[4]
	feature_y_max = result[5]
	feature_x_max = result[6]

	# transform to center / size
	feature_cy = (feature_y_max + feature_y_min) / 2.0 # (num_aboxes,)
	feature_cx = (feature_x_max + feature_x_min) / 2.0 # (num_aboxes,)
	feature_h = feature_y_max - feature_y_min # (num_aboxes,)
	feature_w = feature_x_max - feature_x_min # (num_aboxes,)

	# encode features
	feature_cy = (feature_cy - anchor_boxes[:, 0]) / anchor_boxes[:, 2] / prior_scaling[0] # (num_aboxes,)
	feature_cx = (feature_cx - anchor_boxes[:, 1]) / anchor_boxes[:, 3] / prior_scaling[1] # (num_aboxes,)
	feature_h = tf.log(feature_h / anchor_boxes[:, 2]) / prior_scaling[2] # (num_aboxes,)
	feature_w = tf.log(feature_w / anchor_boxes[:, 3]) / prior_scaling[3] # (num_aboxes,)

	# reorder for ssd
	feature_boxes = tf.stack(
		[feature_cx, feature_cy, feature_w, feature_h],
		axis = -1
	) # (num_aboxes, 4)

	return (
		feature_labels, # (num_aboxes,)
		feature_scores, # (num_aboxes,)
		feature_boxes # (num_aboxes, 4)
	)


def tf_ssd_decode_boxes(localizations, # (N, num_aboxes, 4) <- (cx, cy, w, h)
						anchor_boxes, # (num_aboxes, 4) <- (cy, cx, h, w)
						prior_scaling):
	"""Decode bounding boxes that were computed by the SSD network.

	Arguments:
		localizations: Tensor of shape (N, num_aboxes, 4) containing encoded bounding boxes.
		anchor_boxes: Array of shape (num_aboxes, 4) containing default anchor boxes.
		prior_scaling: List containing scaling of SSD boxes.

	Returns:
		Tensor of shape (N, num_aboxes, 4) containing decoded bounding boxes.
	"""
	# compute center, height and width
	cy = localizations[:, :, 1] * anchor_boxes[:, 2] * prior_scaling[0] + anchor_boxes[:, 0] # (N, num_aboxes)
	cx = localizations[:, :, 0] * anchor_boxes[:, 3] * prior_scaling[1] + anchor_boxes[:, 1] # (N, num_aboxes)
	height = anchor_boxes[:, 2] * tf.exp(localizations[:, :, 3] * prior_scaling[2]) # (N, num_aboxes)
	width = anchor_boxes[:, 3] * tf.exp(localizations[:, :, 2] * prior_scaling[3]) # (N, num_aboxes)

	# boxes coordinates
	ymin = cy - height / 2.0 # (N, num_aboxes)
	xmin = cx - width / 2.0 # (N, num_aboxes)
	ymax = cy + height / 2.0 # (N, num_aboxes)
	xmax = cx + width / 2.0 # (N, num_aboxes)

	return tf.stack([ymin, xmin, ymax, xmax], axis=-1) # (N, num_aboxes, 4) <- (ymin, xmin, ymax, xmax)


def tf_ssd_select_boxes(predictions, # (N, num_aboxes, num_classes)
						localizations, # (N, num_aboxes, 4) <- (ymin, xmin, ymax, xmax)
						confidence_threshold = None,
						top_k = 200):
	"""Select top boxes from predictions using the scores.

	Arguments:
		predictions: Tensor of shape (N, num_aboxes, num_classes) containing the scores of the predicted classes.
		localizations: Tensor of shape (N, num_aboxes, 4) containiing the predicted bounding boxes.
		confidence_threshold: Threshold for filtering out low scores.
		top_k: Number defining how many predictions need to be selected.

	Returns:
		classes: Tensor of shape (N, top_k) containing classes of top boxes.
		scores: Tensor of shape (N, top_k) containing scores of top boxes.
		localizations: Tensor of shape (N, top_k, 4) containing top bounding boxes.
	"""
	# select boxes with confidence threshold
	confidence_threshold = 0.0 if confidence_threshold is None else confidence_threshold

	# select best scores
	predictions = predictions[:, :, 1:] # (N, num_aboxes, num_classes - 1)
	classes = tf.argmax(predictions, axis=2) + 1 # (N, num_aboxes)
	scores = tf.reduce_max(predictions, axis=2) # (N, num_aboxes)
	mask = tf.greater(scores, confidence_threshold) # (N, num_aboxes)
	classes = classes * tf.cast(mask, classes.dtype) # (N, num_aboxes)
	scores = scores * tf.cast(mask, scores.dtype) # (N, num_aboxes)

	# select best classes and bounding boxes
	scores, indexes = tf.nn.top_k(scores, k=top_k, sorted=True) # (N, top_k)
	def gather_fn(classes, localizations, indexes):
		return tf.gather(classes, indexes), tf.gather(localizations, indexes)
	result = tf.map_fn(
		lambda x: gather_fn(x[0], x[1], x[2]),
		(classes, localizations, indexes),
		dtype = (classes.dtype, localizations.dtype),
		parallel_iterations = 10,
		back_prop = False,
		swap_memory = False,
		infer_shape = True
	)

	classes = result[0] # (N, top_k)
	localizations = result[1] # (N, top_k, 4) <- (ymin, xmin, ymax, xmax)

	return (
		classes, # (N, top_k)
		scores, # (N, top_k)
		localizations # (N, top_k, 4) <- (ymin, xmin, ymax, xmax)
	)


def ssd_suppress_overlaps_single(classes, # (top_k,)
								 scores, # (top_k,)
								 localizations, # (top_k, 4) <- (ymin, xmin, ymax, xmax)
								 keep_top_k = 200,
								 nms_threshold = 0.5):
	"""Perform Non-Maximum Suppression on a single sample.

	Arguments:
		classes: Tensor of shape (top_k,) containing top predicted classes.
		scores: Tensor of shape (top_k,) containing top predicted scores.
		localizations: Tensor of shape (top_k, 4) containiing the top predicted bounding boxes.
		keep_top_k: Number defining how many bounding boxes to keep.
		nms_threshold: Threshold for Non-Maximum Suppression.

	Returns:
		classes: Tensor of shape (keep_top_k,) containing classes of top boxes.
		scores: Tensor of shape (keep_top_k,) containing scores of top boxes.
		localizations: Tensor of shape (keep_top_k, 4) containing top bounding boxes.
	"""
	# perform nms
	top_k = tf.shape(classes)[0] # ()
	indexes = tf.image.non_max_suppression(
		localizations,
		scores,
		top_k,
		iou_threshold = nms_threshold
	) # (top_k,)

	# select classes, scores, boxes
	classes = tf.gather(classes, indexes) # (top_k,)
	scores = tf.gather(scores, indexes) # (top_k,)
	localizations = tf.gather(localizations, indexes) # (top_k, 4) <- (ymin, xmin, ymax, xmax)

	# pad tensors to return fixed sized tensors
	pad_size = tf.maximum(keep_top_k - tf.shape(indexes)[0], 0) # ()
	classes = tf.pad(classes, [[0, pad_size]], mode="CONSTANT") # (keep_top_k,)
	scores = tf.pad(scores, [[0, pad_size]], mode="CONSTANT") # (keep_top_k,)
	localizations = tf.pad(localizations, [[0, pad_size], [0, 0]], mode="CONSTANT") # (keep_top_k, 4) <- (ymin, xmin, ymax, xmax)

	return (
		classes, # (keep_top_k,)
		scores, # (keep_top_k,)
		localizations # (keep_top_k, 4) <- (ymin, xmin, ymax, xmax)
	)


def tf_ssd_suppress_overlaps(classes, # (N, top_k)
							 scores, # (N, top_k)
							 localizations, # (N, top_k, 4) <- (ymin, xmin, ymax, xmax)
							 keep_top_k = 200,
							 nms_threshold = 0.5):
	"""Perform Non-Maximum Suppression on a complete batch.

	Arguments:
		classes: Tensor of shape (N, top_k,) containing top predicted classes.
		scores: Tensor of shape (N, top_k,) containing top predicted scores.
		localizations: Tensor of shape (N, top_k, 4) containiing the top predicted bounding boxes.
		keep_top_k: Number defining how many bounding boxes to keep.
		nms_threshold: Threshold for Non-Maximum Suppression.

	Returns:
		classes: Tensor of shape (N, keep_top_k) containing classes of top boxes.
		scores: Tensor of shape (N, keep_top_k) containing scores of top boxes.
		localizations: Tensor of shape (N, keep_top_k, 4) containing top bounding boxes.
	"""
	# process every sample of the batch
	result = tf.map_fn(
		lambda x: ssd_suppress_overlaps_single(x[0], x[1], x[2], keep_top_k=keep_top_k, nms_threshold=nms_threshold),
		(classes, scores, localizations),
		dtype = (classes.dtype, scores.dtype, localizations.dtype),
		parallel_iterations = 10,
		back_prop = False,
		swap_memory = False,
		infer_shape = True
	)

	classes = result[0] # (N, keep_top_k)
	scores = result[1] # (N, keep_top_k)
	localizations = result[2] # (N, keep_top_k, 4) <- (ymin, xmin, ymax, xmax)

	return (
		classes, # (N, keep_top_k)
		scores, # (N, keep_top_k)
		localizations # (N, keep_top_k, 4) <- (ymin, xmin, ymax, xmax)
	)
