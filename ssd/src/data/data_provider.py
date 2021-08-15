##
## /src/data/data_provider.py
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

from utils.common.files import get_full_path
from utils.common.safe_functions import safe_iterator_of_callables


class DataProvider(object):
	"""Handle the complete data input pipeline.

	For each dataset a valid TFRecords file must exist.

	Attributes:
		dataset_name: Name of the dataset to use.
		densify: Flag whether to convert sparse tensors to dense tensors.
		datasets: Dictionary containing the TF dataset and iterator per dataset split.
	"""

	def __init__(self, dataset_name, densify = True):
		"""Initialize the class.

		Arguments:
			dataset_name: Name of the dataset to use.
			densify: Flag whether to convert sparse tensors to dense tensors. Defaults to True.
		"""
		if dataset_name not in ["voc2007", "voc2012", "voc2007+2012"]:
			raise RuntimeError("The dataset '{}' is not supported.".format(dataset_name))

		self.dataset_name = dataset_name
		self.densify = densify

		self.datasets = {}


	def get_tfrecords_path(self, split_name):
		"""Create the full path to the TFRecords file of a dataset split.

		Arguments:
			split_name: Name of the dataset split.

		Returns:
			Full absolute path to the TFRecords file.
		"""
		datasets_name = [self.dataset_name]
		if datasets_name[0] == "voc2007+2012":
			datasets_name = ["voc2007_2012"]

		paths = []
		for dataset in datasets_name:
			path = get_full_path("data", "processed", dataset, "{}.tfrecords".format(split_name))
			paths.append(path)

		return paths


	def init_dataset(self,
					 split_name,
					 batch_size = 0,
					 num_parallel_calls = 2,
					 prefetch_buffer_size = 0,
					 shuffle_buffer_size = 0,
					 shuffle_random_seed = None,
					 preprocessor = None,
					 features = [],
					 padded_shapes = {},
					 name = None):
		"""Initialize the data input pipeline for a dataset split.

		Arguments:
			split_name: Name of the dataset split.
			batch_size: Batch size of the input pipeline. Only used if greater than 1. Defaults to 0.
			num_parallel_calls: Number of calls to fetch data parallel. Defaults to 2.
			prefetch_buffer_size: Buffer size for pre-fetching data. Defaults to 0.
			shuffle_buffer_size: Buffer size for shuffling the data. Defaults to 0.
			shuffle_random_seed: Seed used for random shuffling the data. Defaults to None.
			preprocessor: Object or list of objects that are called in order for preprocessing before batching. Defaults to None.
			features: List of feature keys to extract from the dataset. An empty list extracts all features. Defaults to [].
			padded_shapes: Dictionary of shapes per feature key for padding the batch. Defaults to {}.
			name: Name of the variable scope for this operation. Defaults to 'data_provider'.
		"""
		# check if dataset split was already initialized
		if split_name in self.datasets:
			raise RuntimeError("The dataset '{}' is already initialized.".format(split_name))

		with tf.variable_scope(name, default_name="data_provider"):
			# define redirection for parse map function
			def redirect_to_parse(serialized_example):
				parse_output = self.__parse(
					serialized_example,
					feature_names = features,
					preprocessor = preprocessor
				)
				return parse_output

			# get path to TFRecords file
			tfrecords_path = self.get_tfrecords_path(split_name)

			# open the dataset file and map the features
			dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type="GZIP")
			dataset = dataset.map(
				map_func = redirect_to_parse,
				num_parallel_calls = num_parallel_calls
			)

			# pre-fetch data
			if prefetch_buffer_size > 0:
				dataset = dataset.prefetch(prefetch_buffer_size)

			# shuffle the dataset
			if shuffle_buffer_size > 0:
				dataset = dataset.shuffle(
					shuffle_buffer_size,
					seed = shuffle_random_seed,
					reshuffle_each_iteration = True
				)

			# create batches from the dataset
			if batch_size > 1:
				if len(padded_shapes) > 0:
					dataset = dataset.padded_batch(
						batch_size,
						padded_shapes = padded_shapes
					)
				else:
					dataset = dataset.batch(batch_size)

			# create iterator
			iterator = dataset.make_initializable_iterator()

		# remember dataset and iterator
		self.datasets[split_name] = {
			"dataset": dataset,
			"iterator": iterator,
		}


	def get(self, split_name):
		"""Get tensors defining the output of the data input pipeline.

		Arguments:
			split_name: Name of the dataset split.

		Returns:
			Dictionary of input tensors.
		"""
		if split_name not in self.datasets:
			logging_error("The dataset '{}' is not initialized yet.".format(split_name))

		iterator = self.datasets[split_name]["iterator"]
		return iterator.get_next()


	def get_initializer(self, split_name):
		"""Get initializer to initialize a new run of the dataset split.

		Arguments:
			split_name: Name of the dataset split.

		Returns:
			Operation to initialize a new run of the dataset split.
		"""
		if split_name not in self.datasets:
			logging_error("The dataset '{}' is not initialized yet.".format(split_name))

		return self.datasets[split_name]["iterator"].initializer


	def __parse(self, serialized_example, feature_names = [], preprocessor = None):
		"""Parse and process a single example from the TFRecords file.

		Arguments:
			serialized_example: Example from the TFRecords file.
			feature_names: List of feature keys to extract from the dataset. An empty list extracts all features. Defaults to [].
			preprocessor: Object or list of objects that are called in order for preprocessing before batching. Defaults to None.

		Returns:
			Dictionary containing input tensors.
		"""
		# parse a single example from the TFRecords file
		features = tf.parse_single_example(serialized_example, features={
            "image/filename": tf.FixedLenFeature(shape=(), dtype=tf.string),
            "image/format": tf.FixedLenFeature(shape=(), dtype=tf.string),
            "image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
            "image/width": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "image/height": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "image/channels": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "image/object/bbox/label": tf.VarLenFeature(dtype=tf.int64),
            "image/object/bbox/y_min": tf.VarLenFeature(dtype=tf.float32),
            "image/object/bbox/x_min": tf.VarLenFeature(dtype=tf.float32),
            "image/object/bbox/y_max": tf.VarLenFeature(dtype=tf.float32),
            "image/object/bbox/x_max": tf.VarLenFeature(dtype=tf.float32)
        })

		# replace empty feature names list
		if len(feature_names) == 0:
			feature_names = [
				"image/filename",
	            "image/format",
	            "image/encoded",
	            "image/width",
	            "image/height",
	            "image/channels",
	            "image/object/bbox/label",
	            "image/object/bbox/y_min",
	            "image/object/bbox/x_min",
	            "image/object/bbox/y_max",
	            "image/object/bbox/x_max"
			]

		# densify tensors
		out_features = {}
		for name, tensor in features.items():
			if self.densify and isinstance(tensor, tf.SparseTensor):
				tensor = tf.sparse_tensor_to_dense(tensor)
			out_features[name] = tensor

		# call every pre-processor
		if preprocessor is not None:
			iterator = safe_iterator_of_callables(preprocessor)
			for pp in iterator:
				pp_features = pp(out_features)
				for name, tensor in pp_features.items():
					out_features[name] = tensor

		# create dictionary containing input tensors
		output = {}
		for feature_name in feature_names:
			tensors = out_features[feature_name]
			if isinstance(tensors, list):
				for ii, tensor in enumerate(tensors):
					name = "{}_{}".format(feature_name, ii)
					output[name] = tensor
			else:
				output[feature_name] = tensors
		return output
