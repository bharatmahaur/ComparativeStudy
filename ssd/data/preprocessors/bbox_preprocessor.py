##
## /src/data/preprocessors/bbox_preprocessor.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 07/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from models.ssd.common import tf_ssd_encode_boxes_loop


class BBoxPreprocessor(object):
	"""Handles the pre-processing step that encodes the groundtruth bounding boxes for the forward passes.

	The magic function `__call__(self, *args, **kwargs)` must be implemented in order to function as a valid pre-processing class.

	Attributes:
		anchor_boxes: Array with shape (N, 4) defining the pre-computed default anchor boxes.
		prior_scaling: List or array of length 4 containing scaling parameters for the encoding process.
		matching_threshold: Threshold for IoU when matching the anchor boxes to the groundtruth boxes.
	"""

	def __init__(self, anchor_boxes, prior_scaling, matching_threshold):
		"""Initialize the class.

		Arguments:
			anchor_boxes: Array with shape (N, 4) defining the pre-computed default anchor boxes.
			prior_scaling: List or array of length 4 containing scaling parameters for the encoding process.
			matching_threshold: Threshold for IoU when matching the anchor boxes to the groundtruth boxes.
		"""
		super().__init__()

		self.anchor_boxes = anchor_boxes
		self.prior_scaling = prior_scaling
		self.matching_threshold = matching_threshold


	def __call__(self, inputs):
		"""Handle the pre-processing step defined in this class.

		The following input features are required:
			'image/object/bbox',
			'image/object/bbox/label'.
		The following output features are computed within this step:
			'image/object/encoding/bbox',
			'image/object/encoding/bbox/class',
			'image/object/encoding/bbox/score'.

		Arguments:
			inputs: Dictionary containing all available input features.

		Returns:
			Dictionary containing all input features and the new computed output features.
		"""
		output = {}

		labels, scores, localizations = tf_ssd_encode_boxes_loop(
			inputs["image/object/bbox/label"],
			inputs["image/object/bbox"],
			self.anchor_boxes,
			self.prior_scaling,
			matching_threshold = self.matching_threshold
		)

		output["image/object/encoding/bbox"] = localizations
		output["image/object/encoding/bbox/class"] = labels
		output["image/object/encoding/bbox/score"] = scores

		return output
