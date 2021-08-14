##
## /src/utils/summary.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 15/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import math
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append(sys.path[0])
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models.ssd.common import np_jaccard_score
from utils.common.files import get_full_path


class Summary(object):
	"""Handles easy writing to a summary.

	Attributes:
		session: TF session to handle operations.
		writer: Train writer for the summary.
		__loss_placeholders: Dictionaries containing placeholders for loss.
		__loss_values: Dictionaries containing values for loss.
		__loss_summary_ops: List of summary operations for loss.
		__loss_summary_op: Merged summary operation for loss.
		__aps_placeholders: Dictionaries containing placeholders for average precisions.
		__aps_values: Dictionaries containing values for average precisions.
		__aps_summary_ops: List of summary operations for average precisions.
		__aps_summary_op: Merged summary operation for average precisions.
		__map_placeholder: Placeholder for mAP.
		__map_value: Value for mAP.
		__map_summary_op: Summary operation for mAP.
	"""

	def __init__(self, session, writer):
		"""Initialize the class.

		Arguments:
			session: TF session to handle operations.
			writer: Train writer for the summary.
		"""
		self.session = session
		self.writer = writer

		self.__loss_placeholders = {}
		self.__loss_values = {}
		self.__loss_summary_ops = []
		self.__loss_summary_op = None

		self.__aps_placeholders = {}
		self.__aps_values = {}
		self.__aps_summary_ops = []
		self.__aps_summary_op = None

		self.__map_placeholder = None
		self.__map_value = None
		self.__map_summary_op = None


	def init_losses(self, names = ["confidence", "localization", "l2", "total"]):
		"""Initialize all losses.

		Arguments:
			names: List of loss names. Defaults to ['confidence', 'localization', 'l2', 'total'].
		"""
		for name in names:
			self.init_loss(name)


	def init_loss(self, name):
		"""Initialize a single loss.

		Arguments:
			name: Loss name.
		"""
		base_name = "{}_loss".format(name)
		self.__loss_placeholders[name] = tf.placeholder(tf.float32, name="{}_placeholder".format(base_name))
		self.__loss_values[name] = None
		self.__loss_summary_ops.append(
			tf.summary.scalar(base_name, self.__loss_placeholders[name])
		)


	def set_loss(self, name, value):
		"""Update the value of a loss by name.

		Arguments:
			name: Loss name.
			value: Loss value.
		"""
		self.__loss_values[name] = value


	def init_average_precisions(self, names):
		"""Initialize all average precisions.

		Arguments:
			names: List of average precision names.
		"""
		for name in names:
			self.init_average_precision(name)


	def init_average_precision(self, name):
		"""Initialize a single average precision.

		Arguments:
			name: Average precision name.
		"""
		base_name = "{}_ap".format(name)
		self.__aps_placeholders[name] = tf.placeholder(tf.float32, name="{}_placeholder".format(base_name))
		self.__aps_values[name] = None
		self.__aps_summary_ops.append(
			tf.summary.scalar(base_name, self.__aps_placeholders[name])
		)


	def set_average_precision(self, name, value):
		"""Update the value of an average precision by name.

		Arguments:
			name: Average precision name.
			value: Average precision value.
		"""
		self.__aps_values[name] = value


	def init_mean_average_precision(self):
		"""Initialize mean average precision.
		"""
		self.__map_placeholder = tf.placeholder(tf.float32, name="mAP_placeholder")
		self.__map_value = None
		self.__map_summary_op = tf.summary.scalar("mAP", self.__map_placeholder)


	def set_mean_average_precision(self, value):
		"""Update the value of the mean average precision.

		Arguments:
			value: Mean average precision value.
		"""
		self.__map_value = value


	def merge(self):
		"""Merge all summary operations.
		"""
		self.__loss_summary_op = tf.summary.merge(self.__loss_summary_ops)
		self.__aps_summary_op = tf.summary.merge(self.__aps_summary_ops)


	def write_loss(self, global_step):
		"""Write losses to the summary file.

		Arguments:
			global_step: The step value associated with the losses.
		"""
		# create dictionary for placeholders with values
		feed_dict = {}
		for name, placeholder in self.__loss_placeholders.items():
			value = self.__loss_values[name]
			if value is not None:
				feed_dict[placeholder] = value
		if len(feed_dict) == 0:
			return

		# write values to summary
		summary = self.session.run(self.__loss_summary_op, feed_dict=feed_dict)
		self.writer.add_summary(summary, global_step)
		self.writer.flush()

		# reset values
		for name in self.__loss_values:
			self.__loss_values[name] = []


	def write_average_precision(self, global_step):
		"""Write average precisions to the summary file.

		Arguments:
			global_step: The step value associated with the average precisions.
		"""
		# create dictionary for placeholders with values
		feed_dict = {}
		for name, placeholder in self.__aps_placeholders.items():
			value = self.__aps_values[name]
			if value is not None:
				feed_dict[placeholder] = value
		if len(feed_dict) == 0:
			return

		# write values to summary
		summary = self.session.run(self.__aps_summary_op, feed_dict=feed_dict)
		self.writer.add_summary(summary, global_step)
		self.writer.flush()

		# reset values
		for name in self.__aps_values:
			self.__aps_values[name] = None


	def write_mean_average_precision(self, global_step):
		"""Write mean average precision to the summary file.

		Arguments:
			global_step: The step value associated with the mean average precision.
		"""
		# create dictionary for placeholder with value
		feed_dict = {}
		if self.__map_value is not None:
			feed_dict[self.__map_placeholder] = self.__map_value
		if len(feed_dict) == 0:
			return

		# write value to summary
		summary = self.session.run(self.__map_summary_op, feed_dict=feed_dict)
		self.writer.add_summary(summary, global_step)
		self.writer.flush()

		# reset value
		self.__map_value = None
