##
## /src/data/preprocessors/image_preprocessor.py
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

from utils.preprocessing import random_crop_image, random_distort_colors, random_expand_image, random_flip_left_right


class ImagePreprocessor(object):
	"""Handle the pre-processing step that resizes an image and apply data augmentation on it.

	The magic function `__call__(self, *args, **kwargs)` must be implemented in order to function as a valid pre-processing class.

	Attributes:
		image_shape: a tuple defining the output shape of the image.
		op_seed: the seed used for random operations.
		data_augmentation: a flag whether to apply data augmentation.
	"""

	def __init__(self, image_shape, op_seed = None, data_augmentation = True):
		"""Initializes the class.

		Arguments:
			image_shape: Tuple defining the output shape of the image.
			op_seed: Seed used for random operations. Defaults to None.
			data_augmentation: Flag whether to apply data augmentation. Defaults to True.
		"""
		super().__init__()

		self.image_shape = image_shape
		self.op_seed = op_seed
		self.data_augmentation = data_augmentation


	def __call__(self, inputs):
		"""Handle the pre-processing step defined in this class.

		The following input features are required:
			'image',
			'image/object/bbox',
			'image/object/bbox/label'.
		The following output features are computed within this step:
			'image',
			'image/object/bbox',
			'image/object/bbox/label'.

		Arguments:
			inputs: Dictionary containing all available input features.

		Returns:
			Dictionary containing all input features and the new computed output features.
		"""
		output = {}

		image = inputs["image"]
		boxes = inputs["image/object/bbox"]
		labels = inputs["image/object/bbox/label"]

		# convert image to [0.0, 1.0]
		image = tf.divide(image, 255.0)

		if self.data_augmentation:
			# distort colors
			image = random_distort_colors(image, seed=self.op_seed)

			# reorder channels
			channel_indexes = tf.random_shuffle([0, 1, 2], seed=self.op_seed)
			image = tf.gather(image, channel_indexes, axis=-1)

			# expand image
			rgb_mean = [123.0 / 255.0, 117.0 / 255.0, 104.0 / 255.0]
			image, boxes = random_expand_image(image, boxes, mean_value=rgb_mean, seed=self.op_seed)

			# distort bounding boxes
			def crop_image(image, boxes, labels, seed, min_object_covered):
				image, boxes, labels = random_crop_image(
					image,
					boxes,
					labels,
					seed = seed,
					min_object_covered = min_object_covered
				)
				return image, boxes, labels
			case_value = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32, seed=self.op_seed)
			image, boxes, labels = tf.case({
				tf.equal(case_value, 0): lambda: crop_image(image, boxes, labels, self.op_seed, 0.3),
				tf.equal(case_value, 1): lambda: crop_image(image, boxes, labels, self.op_seed, 0.5),
				tf.equal(case_value, 2): lambda: crop_image(image, boxes, labels, self.op_seed, 0.7)
			}, exclusive=True)

			# flip image
			image, boxes = random_flip_left_right(image, boxes, seed=self.op_seed)

		# resize image
		image = tf.image.resize_images(
			image,
			self.image_shape,
			method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
			align_corners = False
		)

		# convert image back to [0.0, 255.0]
		image = tf.multiply(image, 255.0)

		output["image"] = image
		output["image/object/bbox"] = boxes
		output["image/object/bbox/label"] = labels

		return output
