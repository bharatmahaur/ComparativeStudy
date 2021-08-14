##
## /src/data/preprocessors/default_preprocessor.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 15/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


class DefaultPreprocessor(object):
	"""Handle the pre-processing step that decodes raw image data from the features.

	The magic function `__call__(self, *args, **kwargs)` must be implemented in order to function as a valid pre-processing class.
	"""

	def __init__(self):
		"""Initializes the class.
		"""
		super().__init__()


	def __call__(self, inputs):
		"""Handle the pre-processing step defined in this class.

		The following input features are required:
			'image/format',
			'image/encoded',
			'image/{height,width,channels}',
			'image/object/bbox/{y_min,x_min,y_max,x_max}'.
		The following output features are computed within this step:
			'image',
			'image/shape',
			'image/object/bbox'.

		Arguments:
			inputs: Dictionary containing all available input features.

		Returns:
			Dictionary containing all input features and the new computed output features.
		"""
		output = {}

		# decode image
		image = tf.cond(
			tf.equal(inputs["image/format"], tf.constant("jpeg", dtype=tf.string)),
			true_fn = lambda image=inputs["image/encoded"]: tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_ACCURATE"),
			false_fn = lambda image=inputs["image/encoded"]: tf.image.decode_image(image, channels=3)
		)
		image.set_shape((None, None, 3))

		# build image shape
		image_shape = tf.stack([inputs["image/height"], inputs["image/width"], inputs["image/channels"]])

		# build bounding boxes
		image_object_bbox = tf.stack([
			inputs["image/object/bbox/y_min"],
			inputs["image/object/bbox/x_min"],
			inputs["image/object/bbox/y_max"],
			inputs["image/object/bbox/x_max"]
		], axis=1)

		# change image dtype to float
		if image.dtype != tf.float32:
			image = tf.cast(image, tf.float32)

		output["image"] = image
		output["image/shape"] = image_shape
		output["image/object/bbox"] = image_object_bbox

		return output
