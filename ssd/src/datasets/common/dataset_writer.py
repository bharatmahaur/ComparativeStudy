##
## /src/datasets/common/dataset_writer.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 13/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import numpy as np
import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.files import mkdir


class DatasetWriter(object):
	"""Handle simple writing to TFRecords files.

	Attributes:
		path: TFrecords file path.
		compression_type: Compression of the TFRecords file.
		writer: Writer for TFRecords files.
	"""

	def __init__(self, path, compression_type = tf.python_io.TFRecordCompressionType.GZIP):
		"""Initialize the class.

		Arguments:
			path: TFrecords file path.
			compression_type: Compression of the TFRecords file. Defaults to tf.python_io.TFRecordCompressionType.GZIP.
		"""
		self.path = path
		if not self.path.endswith(".tfrecords"):
			self.path = "{}.tfrecords".format(self.path)
		self.compression_type = compression_type

		self.writer = None


	def open(self):
		"""Open the TFRecords file.
		"""
		parent_path = os.path.dirname(self.path)
		mkdir(parent_path)

		# open the TFRecords file
		options = None
		if self.compression_type is not None:
			options = tf.python_io.TFRecordOptions(self.compression_type)
		self.writer = tf.python_io.TFRecordWriter(self.path, options=options)


	def close(self):
		"""Close the TFRecords file.
		"""
		self.writer.close()
		self.writer = None


	def write_single_example(self, features):
		"""Write a single sample of features to the TFRecords file.

		Arguments:
			features: Dictionary containing features.
		"""
		example = tf.train.Example(features=tf.train.Features(feature=features))
		self.writer.write(example.SerializeToString())


	def int64_feature(self, value):
		"""Convert an integer to a TFRecords compatible feature.

		Arguments:
			value: Integer or list of integers.

		Returns:
			TFRecords compatible feature.
		"""
		if not isinstance(value, (list, tuple)):
			value = [value]
		return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


	def float_feature(self, value):
		"""Convert a float to a TFRecords compatible feature.

		Arguments:
			value: Float or list of floats.

		Returns:
			TFRecords compatible feature.
		"""
		if not isinstance(value, (list, tuple)):
			value = [value]
		return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


	def bytes_feature(self, value):
		"""Convert bytes to a TFRecords compatible feature.

		Arguments:
			value: Bytes or list of bytes.

		Returns:
			TFRecords compatible feature.
		"""
		if not isinstance(value, (list, tuple)):
			value = [value]
		value_bytes = []
		for v in value:
			if isinstance(v, bytes):
				value_bytes.append(v)
				continue
			value_bytes.append(tf.compat.as_bytes(v))
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(value_bytes)))


	def dtype_feature(self, value):
		"""Convert an array to a TFRecords compatible feature.

		Arguments:
			value: Float or integer array.

		Returns:
			TFRecords compatible feature.
		"""
		if value.dtype == np.float32 or value.dtype == np.float64:
			return tf.train.Feature(float_list=tf.train.FloatList(value=value))
		elif value.dtype == np.int64:
			return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
		raise TypeError("The dtype '{}' is not supported.".format(value.dtype))
