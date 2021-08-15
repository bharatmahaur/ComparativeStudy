##
## /src/datasets/common/dataset_source.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 13/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import numpy as np
import os
import shutil
import sys
import tensorflow as tf
from abc import ABC, abstractmethod
from tqdm import tqdm

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.archives import extract_archive
from utils.common.files import get_full_path, get_directories
from utils.common.logging import logging, logging_info, logging_warn


class DatasetSource(ABC):
	"""Abstract class for handling dataset sources.

	Attributes:
		dataset: Name of the dataset.
		compression_type: Compression of the TFRecords file.

		paths: Dictionary containing absolute paths to dataset directories.
	"""

	def __init__(self, dataset, compression_type = tf.python_io.TFRecordCompressionType.GZIP):
		"""Initialize the class.

		Arguments:
			dataset: Name of the dataset.
			compression_type: Compression of the TFRecords file. Defaults to tf.python_io.TFRecordCompressionType.GZIP.
		"""
		self.dataset = dataset
		self.compression_type = compression_type

		self.paths = {
			"raw": get_full_path("data", "raw", dataset),
			"interim": get_full_path("data", "interim", dataset),
			"processed": get_full_path("data", "processed", dataset)
		}


	@property
	@abstractmethod
	def labels(self):
		"""List containing all available labels.
		"""
		pass


	@property
	@abstractmethod
	def selected_labels(self):
		"""List containing selected labels for the dataset.
		"""
		pass


	@property
	def num_classes(self):
		"""Number of selected labels for the dataset.
		"""
		return len(self.selected_labels)


	def get_supported_archive_extensions(self):
		"""Archive extensions that are supported for extracting.

		Returns:
			Tuple of supported archive extensions.
		"""
		return (".zip", ".tar", ".tar.bz2", ".tar.gz")


	def get_archives(self, include_path = False):
		"""Get paths of supported archives.

		Arguments:
			include_path: Flag whether to include the absolute path. Defaults to False.

		Returns:
			List of supported archives.
		"""
		path = os.path.join(self.paths["raw"])
		items = os.listdir(path)
		extensions = self.get_supported_archive_extensions()
		archives = []

		# iterate through directory contents
		for item in items:
			if not item.endswith(extensions):
				continue
			if include_path:
				archives.append(os.path.join(path, item))
			else:
				archives.append(item)

		return archives


	def get_archive_details(self):
		"""Get path details of supported archives.

		Returns:
			Dictionary containg details like name, extension and absolute path of supported archives.
		"""
		path = os.path.join(self.paths["raw"])
		items = os.listdir(path)
		extensions = self.get_supported_archive_extensions()
		archives = {}

		# iterate through directory contents
		for item in items:
			for extension in extensions:
				if item.endswith(extension):
					name = item[:-len(extension)]
					archives[name] = {
						"name": item,
						"extension": extension,
						"path": os.path.join(path, item)
					}
					break

		return archives


	def get_extracted_archives(self):
		"""Get list of extracted archives.

		Returns:
			List of extracted archives.
		"""
		return get_directories(self.paths["interim"])


	def extract(self, overwrite = False):
		"""Extract all supported archives of a source.

		Arguments:
			overwrite: Flag whether to overwrite already extracted archives. Defaults to False.
		"""
		archives = self.get_archive_details()
		if len(archives) == 0:
			logging_warn("No supported archives for the dataset '{}' could be found.".format(self.dataset))
			return

		# iterate through supported archives
		for name, params in archives.items():
			target_path = os.path.join(self.paths["interim"], name)

			# skip or delete already extracted archive
			if os.path.isdir(target_path):
				if overwrite:
					logging_info("Archive '{}' already extracted. Deleting".format(params["name"]), end=" ... ")
					shutil.rmtree(target_path)
					logging("ok")
				else:
					logging_info("Archive '{}' already extracted. Skip.".format(params["name"]))
					continue

			# extract archive
			logging_info("Extracting archive '{}'".format(params["name"]), end=" ... ")
			extract_archive(params["path"], self.paths["interim"])
			logging("ok")


	@property
	def supported_image_formats(self):
		"""Dictionary of supported image formats. Values are tuples containing file extensions.
		"""
		return {
			"bmp": (".bmp", ".dib"),
			"jpeg": (".jfi", ".jfif", ".jif", ".jpe", ".jpeg", ".jpg"),
			"png": (".png",)
		}


	@property
	def supported_image_extensions(self):
		"""Tuple of supported image extensions.
		"""
		formats = self.supported_image_formats
		extensions = ()
		for values in formats.values():
			extensions = extensions + values
		return extensions


	def get_image_format_of_file(self, file):
		"""Get image format of a given file.

		Arguments:
			file: Name of the file.

		Returns:
			Image format of the file.
		"""
		file = file.lower()
		for format, extensions in self.supported_image_formats.items():
			if file.endswith(extensions):
				return format
		return None


	@abstractmethod
	def convert(self, split_name):
		"""Convert data of a given split into a TFRecords file.

		Arguments:
			split_name: Name of the dataset split to convert.
		"""
		pass


	def calculate_rgb_mean(self, split_name):
		"""Calculate the RGB mean of all images in a dataset split. The split must be converted to a TFRecords file.

		Arguments:
			split_name: Name of the dataset split.

		Returns:
			Array of length 3 containing mean values of RGB channels.
		"""
		dataset_path = os.path.join(self.paths["processed"], "{}.tfrecords".format(split_name))
		if not os.path.isfile(dataset_path):
			logging_warn("There is no TFRecords file for the split '{}'.".format(split_name))

		# initialize reader for TFRecords files
		filename_queue = tf.train.string_input_producer([dataset_path], num_epochs=1)

		options = None
		if self.compression_type is not None:
			options = tf.python_io.TFRecordOptions(self.compression_type)
		reader = tf.TFRecordReader(options=options)

		# read image and format from file
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_example, features={
			"image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
			"image/format": tf.FixedLenFeature(shape=(), dtype=tf.string)
		})

		# setup decoder
		decode_bmp_fn = lambda: tf.image.decode_bmp(features["image/encoded"], channels=3)
		decode_jpeg_fn = lambda: tf.image.decode_jpeg(features["image/encoded"], channels=3, dct_method="INTEGER_ACCURATE")
		decode_png_fn = lambda: tf.image.decode_png(features["image/encoded"], channels=3, dtype=tf.uint8)
		decoded_image = tf.case({
			tf.equal(features["image/format"], "bmp"): decode_bmp_fn,
			tf.equal(features["image/format"], "jpeg"): decode_jpeg_fn,
			tf.equal(features["image/format"], "png"): decode_png_fn
		}, exclusive=True)

		# set minimum logging level for TensorFlow
		old_min_log_level = os.environ["TF_CPP_MIN_LOG_LEVEL"] if "TF_CPP_MIN_LOG_LEVEL" in os.environ else ""
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

		with tf.Session() as session:
			session.run([
				tf.global_variables_initializer(),
				tf.local_variables_initializer()
			])

			# start loading images
			coordinator = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

			# decode images
			logging_info("Loading images from TFRecords file ...")
			num_samples = np.sum([1 for _ in tf.python_io.tf_record_iterator(dataset_path, options=options)])
			images = []
			for ii in tqdm(range(num_samples), ascii=True):
				image = session.run(decoded_image)
				images.append(image)

			coordinator.request_stop()
			coordinator.join(threads)

		# reset minimum log level for TensorFlow
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = old_min_log_level

		# calculate RGB mean of images
		logging_info("Calculating RGB mean of dataset split ...")
		rgb_mean = np.zeros((3,), dtype=np.uint8)
		num_pixels = 0
		for image in tqdm(images, ascii=True):
			rgb_mean = np.add(rgb_mean, np.sum(image, axis=(0, 1)))
			num_pixels += image.shape[0] * image.shape[1]

		rgb_mean = np.divide(rgb_mean, num_pixels)
		return rgb_mean
