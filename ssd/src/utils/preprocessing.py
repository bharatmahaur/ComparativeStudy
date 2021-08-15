##
## /src/utils/preprocessing.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


def random_distort_colors(image, # (h, w, c)
						  seed = None):
	"""Randomly distort the colors of an image by applying brightness, saturation, hue, contrast. The order of the distortions is
	randomly chosen.

	Arguments:
		image: Image to process.
		seed: Seed to use for random operations. Defaults to None.

	Returns:
		Distorted image.
	"""
	def distort_0(image, seed = None):
		image = tf.image.random_brightness(image, 0.13, seed=seed)
		image = tf.image.random_saturation(image, 0.5, 1.5, seed=seed)
		image = tf.image.random_hue(image, 0.2, seed=seed)
		image = tf.image.random_contrast(image, 0.5, 1.5, seed=seed)
		return tf.clip_by_value(image, 0.0, 1.0)

	def distort_1(image, seed = None):
		image = tf.image.random_saturation(image, 0.5, 1.5, seed=seed)
		image = tf.image.random_brightness(image, 0.13, seed=seed)
		image = tf.image.random_contrast(image, 0.5, 1.5, seed=seed)
		image = tf.image.random_hue(image, 0.2, seed=seed)
		return tf.clip_by_value(image, 0.0, 1.0)

	def distort_2(image, seed = None):
		image = tf.image.random_contrast(image, 0.5, 1.5, seed=seed)
		image = tf.image.random_hue(image, 0.2, seed=seed)
		image = tf.image.random_brightness(image, 0.13, seed=seed)
		image = tf.image.random_saturation(image, 0.5, 1.5, seed=seed)
		return tf.clip_by_value(image, 0.0, 1.0)

	def distort_3(image, seed = None):
		image = tf.image.random_hue(image, 0.2, seed=seed)
		image = tf.image.random_saturation(image, 0.5, 1.5, seed=seed)
		image = tf.image.random_contrast(image, 0.5, 1.5, seed=seed)
		image = tf.image.random_brightness(image, 0.13, seed=seed)
		return tf.clip_by_value(image, 0.0, 1.0)

	# randomly choose the order of distortions
	case_value = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32, seed=seed)
	return tf.case({
		tf.equal(case_value, 0): lambda: distort_0(image, seed=seed),
		tf.equal(case_value, 1): lambda: distort_1(image, seed=seed),
		tf.equal(case_value, 2): lambda: distort_2(image, seed=seed),
		tf.equal(case_value, 3): lambda: distort_3(image, seed=seed)
	}, exclusive=True)


def random_expand_image(image, # (h, w, c)
						boxes, # (num_boxes, 4) <- (y_min, x_min, x_max, y_max)
						max_ratio = 4.0,
						mean_value = [0.0, 0.0, 0.0],
						seed = None):
	"""Randomly expand image by placing the image on a bigger canvas to reduce the size of the image and its objects.

	Arguments:
		image: Image to process.
		boxes: Bounding boxes of the sample.
		max_ratio: Maximum ratio of the bigger canvas in relation to the image. Defaults to 4.0
		mean_value: RGB values to fill the canvas with. Defaults to [0, 0, 0].
		seed: Seed to use for random operations. Defaults to None.

	Returns:
		expanded_image: Expanded image including the canvas.
		boxes: Transformed bounding boxes.
	"""
	# choose random ratio for the canvas
	ratio = tf.random_uniform([], minval=1.0, maxval=max_ratio, dtype=tf.float32, seed=seed)

	# compute sizes
	image_height = tf.shape(image)[0]
	image_width = tf.shape(image)[1]

	fimage_height = tf.cast(image_height, tf.float32)
	fimage_width = tf.cast(image_width, tf.float32)

	expanded_height = tf.cast(ratio * fimage_height, tf.int32)
	expanded_width = tf.cast(ratio * fimage_width, tf.int32)

	fexpanded_height = tf.cast(expanded_height, tf.float32)
	fexpanded_width = tf.cast(expanded_width, tf.float32)

	# compute size of image padding to place image randomly on canvas
	pad_height = expanded_height - image_height
	pad_width = expanded_width - image_width

	top_pad_size = tf.random_uniform([], minval=0, maxval=pad_height + 1, dtype=tf.int32, seed=seed)
	left_pad_size = tf.random_uniform([], minval=0, maxval=pad_width + 1, dtype=tf.int32, seed=seed)
	bottom_pad_size = pad_height - top_pad_size
	right_pad_size = pad_width - left_pad_size

	paddings = tf.stack([
		[top_pad_size, bottom_pad_size],
		[left_pad_size, right_pad_size],
		[0, 0]
	], axis=0)

	image = image - mean_value
	expanded_image = tf.pad(image, paddings, mode="CONSTANT", constant_values=0)
	expanded_image = expanded_image + mean_value

	# transform boxes
	boxes = tf.multiply(boxes, tf.stack([
		fimage_height,
		fimage_width,
		fimage_height,
		fimage_width
	]))
	boxes = tf.add(boxes, tf.stack([
		tf.cast(top_pad_size, tf.float32),
		tf.cast(left_pad_size, tf.float32),
		tf.cast(top_pad_size, tf.float32),
		tf.cast(left_pad_size, tf.float32)
	]))
	boxes = tf.divide(boxes, tf.stack([
		fexpanded_height,
		fexpanded_width,
		fexpanded_height,
		fexpanded_width
	]))

	return expanded_image, boxes


def random_crop_image(image, # (h, w, c)
					  boxes, # (N, 4)
					  labels, # (N,)
					  seed = None,
					  min_object_covered = 0.5,
					  aspect_ratio_range = (0.5, 2.0),
					  area_range = (0.1, 1.0),
					  max_attempts = 200):
	"""Randomly crop an image and its bounding boxes.

	Arguments:
		image: Image to process.
		boxes: Bounding boxes of the sample.
		labels: Labels of the bounding boxes.
		seed: Seed to use for random operations. Defaults to None.
		min_object_covered: Minimum fraction of any bounding box that is included in the result image. Defaults to 0.5.
		aspect_ratio_range: Range of aspect ratios of the result image. Defaults to (0.5, 2.0).
		area_range: Range of area in relation to the image of the result image. Defaults to (0.1, 1.0).
		max_attempts: Maximum number of attempts to find a cropping. Defaults to 200.

	Returns:
		cropped_image: Result image.
		boxes: Transformed boxes.
		labels: Transformed labels.
	"""
	def tf_intersection(box, # (4,)
						boxes): # (N, 4)
		"""Compute intersection score of a single bounding box to other boxes.

		Arguments:
			box: Array containing coordinates of the reference bounding box.
			boxes: Array containing coordinates of the multiple bounding boxes.

		Returns:
			Array containing intersection score of reference box to the other boxes.
		"""
		boxes = tf.transpose(boxes) # (4, N)
		box = tf.transpose(box) # (4,)

		y_min = tf.maximum(boxes[0], box[0]) # (N,)
		x_min = tf.maximum(boxes[1], box[1]) # (N,)
		y_max = tf.minimum(boxes[2], box[2]) # (N,)
		x_max = tf.minimum(boxes[3], box[3]) # (N,)

		intersection_volumes = tf.maximum(y_max - y_min, 0.0) * tf.maximum(x_max - x_min, 0.0) # (N,)
		boxes_volumes = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1]) # (N,)
		return tf.divide(intersection_volumes, boxes_volumes) # (N,)

	# distort bounding boxes
	box_begin, box_size, distort_box = tf.image.sample_distorted_bounding_box(
		tf.shape(image),
		tf.expand_dims(boxes, 0),
		seed = seed,
		min_object_covered = min_object_covered,
		aspect_ratio_range = aspect_ratio_range,
		area_range = area_range,
		max_attempts = max_attempts,
		use_image_if_no_bounding_boxes = True
	)
	distort_box = distort_box[0, 0]

	# crop image
	cropped_image = tf.slice(image, box_begin, box_size)
	cropped_image.set_shape([None, None, 3])

	# resize bounding boxes (translate + scale)
	boxes = tf.subtract(boxes, tf.stack([
		distort_box[0],
		distort_box[1],
		distort_box[0],
		distort_box[1]
	]))
	boxes = tf.divide(boxes, tf.stack([
		distort_box[2] - distort_box[0],
		distort_box[3] - distort_box[1],
		distort_box[2] - distort_box[0],
		distort_box[3] - distort_box[1]
	]))

	# filter out overlapping bounding boxes
	scores = tf_intersection(
		tf.constant([0, 0, 1, 1], boxes.dtype),
		boxes
	)
	mask = tf.greater(scores, min_object_covered)
	labels = tf.boolean_mask(labels, mask)
	boxes = tf.boolean_mask(boxes, mask)

	# adjust boxes to [0.0, 1.0]
	y_min = tf.maximum(boxes[:, 0], 0.0)
	x_min = tf.maximum(boxes[:, 1], 0.0)
	y_max = tf.minimum(boxes[:, 2], 1.0)
	x_max = tf.minimum(boxes[:, 3], 1.0)
	boxes = tf.stack([y_min, x_min, y_max, x_max], axis = -1)

	return cropped_image, boxes, labels


def random_flip_left_right(image, boxes, seed = None):
	"""Randomly flip an image horizontally.

	Arguments:
		image: Image to process.
		boxes: Bounding boxes of the sample.
		seed: Seed to use for random operations. Defaults to None.

	Returns:
		image: Flipped image.
		boxes: Flipped bounding boxes.
	"""
	def boxes_flip_left_right(boxes):
		"""Flip bounding boxes horizontally.

		Arguments:
			boxes: Bounding boxes to flip.

		Returns:
			Flipped bounding boxes.
		"""
		return tf.stack([
			boxes[:, 0],
			1.0 - boxes[:, 3],
			boxes[:, 2],
			1.0 - boxes[:, 1]
		], axis=-1)

	# get probability of flipping the image and its boxes
	uniform_random = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
	condition = tf.less(uniform_random, 0.5)

	# flip image
	image = tf.cond(
		condition,
		lambda: tf.image.flip_left_right(image),
		lambda: image
	)

	# flip bounding boxes
	boxes = tf.cond(
		condition,
		lambda: boxes_flip_left_right(boxes),
		lambda: boxes
	)

	return image, boxes
