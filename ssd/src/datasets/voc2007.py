##
## /src/datasets/voc2007.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 13/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import argparse
import os
import sys
import tensorflow as tf
import xml.etree.ElementTree
from datetime import datetime
from tqdm import tqdm

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from datasets.common import DatasetSource, DatasetWriter
from utils.common.logging import logging_info


# wget -c "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
# wget -c "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
class VOC2007(DatasetSource):
	"""Handle the Pascal VOC 2007 dataset.
	"""

	def __init__(self):
		"""Initialize the class.
		"""
		super().__init__("voc2007")


	@property
	def labels(self):
		return [
			"aeroplane",
			"bicycle",
			"bird",
			"boat",
			"bottle",
			"bus",
			"car",
			"cat",
			"chair",
			"cow",
			"diningtable",
			"dog",
			"horse",
			"motorbike",
			"person",
			"pottedplant",
			"sheep",
			"sofa",
			"train",
			"tvmonitor"
		]


	@property
	def selected_labels(self):
		# select all labels
		return self.labels


	def convert(self, split_name):
		if split_name not in ["train", "validation", "test"]:
			raise NotImplementedError("The dataset split '{}' is currently not supported.".format(split_name))

		# open the dataset writer
		destination_path = os.path.join(self.paths["processed"], split_name)
		writer = DatasetWriter(destination_path, compression_type=self.compression_type)
		writer.open()

		# convert the split
		self.__convert_split(split_name, writer)

		# close the dataset writer
		writer.close()


	def __convert_split(self, split_name, writer):
		"""Convert a single dataset split.

		Arguments:
			split_name: Name of the dataset split.
			devkit_path: Path to the VOC devkit.
			writer: TFRecords file writer.
		"""
		if split_name in ["train", "validation"]:
			devkit_path = os.path.join(self.paths["interim"], "VOCtrainval_06-Nov-2007", "VOCdevkit", "VOC2007")
		elif split_name == "test":
			devkit_path = os.path.join(self.paths["interim"], "VOCtest_06-Nov-2007", "VOCdevkit", "VOC2007")

		# load image ids
		if split_name in ["train", "test"]:
			imageset_filename = "{}.txt".format(split_name)
		elif split_name == "validation":
			imageset_filename = "val.txt"
		path = os.path.join(devkit_path, "ImageSets", "Main", imageset_filename)
		with open(path, "r") as file:
			image_ids = file.readlines()
		image_ids = [image_id.strip() for image_id in image_ids]

		# process annotations
		logging_info("Processing and writing images ...")

		# iterate all images
		num_images = 0
		annotations_path = os.path.join(devkit_path, "Annotations")
		images_path = os.path.join(devkit_path, "JPEGImages")
		for image_id in tqdm(image_ids, ascii=True):
			# meta information
			meta_url = "host.robots.ox.ac.uk"
			meta_time = datetime.utcnow().isoformat()
			meta_requester = "paul@warkentin.email"

			# read xml file
			xml_path = os.path.join(annotations_path, "{}.xml".format(image_id))
			if not os.path.isfile(xml_path):
				continue
			xml_root = xml.etree.ElementTree.parse(xml_path).getroot()

			# check format of image
			filename = xml_root.findtext("filename")
			image_format = self.get_image_format_of_file(filename)
			if image_format is None:
				raise NotImplementedError("The format of the file '{}' is currently not supported.".format(filename))

			# read image size
			image_height = int(xml_root.find("size").findtext("height"))
			image_width = int(xml_root.find("size").findtext("width"))
			image_channels = int(xml_root.find("size").findtext("depth"))

			# read image
			image_path = os.path.join(images_path, filename)
			with tf.gfile.FastGFile(image_path, "rb") as file:
				image_raw_data = file.read()

			# read bounding boxes
			labels = []
			bboxes = [[], [], [], []]
			for sobject in xml_root.findall("object"):
				label_name = sobject.find("name").text
				if label_name not in self.selected_labels:
					continue
				labels.append(self.selected_labels.index(label_name) + 1)
				bndbox = sobject.find("bndbox")
				bboxes[0].append(int(bndbox.find("ymin").text) / image_height)
				bboxes[1].append(int(bndbox.find("xmin").text) / image_width)
				bboxes[2].append(int(bndbox.find("ymax").text) / image_height)
				bboxes[3].append(int(bndbox.find("xmax").text) / image_width)
			if len(labels) == 0:
				continue

			# write sample
			writer.write_single_example({
				"meta/url": writer.bytes_feature(meta_url),
				"meta/requester": writer.bytes_feature(meta_requester),
				"meta/time": writer.bytes_feature(meta_time),
				"image/filename": writer.bytes_feature(filename),
				"image/format": writer.bytes_feature(image_format),
				"image/encoded": writer.bytes_feature(image_raw_data),
				"image/width": writer.int64_feature(image_width),
				"image/height": writer.int64_feature(image_height),
				"image/channels": writer.int64_feature(image_channels),
				"image/shape": writer.int64_feature((image_height, image_width, image_channels)),
				"image/object/bbox/label": writer.int64_feature(labels),
				"image/object/bbox/y_min": writer.float_feature(bboxes[0]),
				"image/object/bbox/x_min": writer.float_feature(bboxes[1]),
				"image/object/bbox/y_max": writer.float_feature(bboxes[2]),
				"image/object/bbox/x_max": writer.float_feature(bboxes[3])
			})
			num_images += 1

		logging_info("Successfully written {} image(s) to the TFRecords file.".format(num_images))


if __name__ == "__main__":

	# initialize arguments
	parser = argparse.ArgumentParser(
		description = "PASCAL Visual Object Classes 2007"
	)
	parser.add_argument(
		"--extract",
		action = "store_true",
		help = "Extract all archives."
	)
	parser.add_argument(
		"--convert",
		type = str,
		choices = ["train", "validation", "test"],
		help = "Convert a dataset split to TFRecords. The data must be extracted."
	)
	parser.add_argument(
		"--rgb-mean",
		type = str,
		choices = ["train", "validation", "test"],
		help = "Calculate the RGB mean of a dataset split. The split must be converted to TFRecords."
	)
	arguments = parser.parse_args()

	if not arguments.extract and arguments.convert is None and arguments.rgb_mean is None:
		parser.print_help()
		exit()

	# initialize dataset
	dataset = VOC2007()

	# extract dataset
	if arguments.extract:
		logging_info("Extract all archives.")
		dataset.extract(overwrite=True)

	# convert dataset
	if arguments.convert is not None:
		logging_info("Convert dataset split '{}'.".format(arguments.convert))
		dataset.convert(arguments.convert)

	# calculate rgb mean
	if arguments.rgb_mean is not None:
		logging_info("Calculate RGB mean of the dataset split '{}'.".format(arguments.rgb_mean))
		rgb_mean = dataset.calculate_rgb_mean(arguments.rgb_mean)
		logging_info("RGB mean is R = {:.2f}, G = {:.2f}, B = {:.2f}.".format(*rgb_mean))
