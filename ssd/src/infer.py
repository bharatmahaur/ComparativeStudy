##
## /src/models/infer.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 16/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from datasets.common.utils import get_dataset
from utils import ArgumentList, Run
from utils.common.files import get_full_path
from utils.common.logging import logging_info
from utils.common.terminal import query_yes_no
from utils.tfu import tfu_load_graph, tfu_set_logging
from utils.visualization import draw_single_box, get_distinct_colors


if __name__ == "__main__":

	# parse arguments
	argument_list = ArgumentList(
		description = "Apply an exported model on a single image."
	)
	argument_list.add_image_filename_argument("The input image filename.", required=True)
	argument_list.add_model_argument("The model used for training.", default=None, required=True)
	argument_list.add_model_name_argument("The exported model name.", required=True)
	argument_list.add_dataset_argument("The dataset used for training.", default=None)
	argument_list.add_tf_verbosity_argument("Tensorflow verbosity.", default="info")
	argument_list.add_tf_min_log_level_argument("Tensorflow minimum log level.", default=3)
	arguments = argument_list.parse()

	# print some information
	logging_info("Image filename:               {}".format(arguments.image_filename))
	logging_info("Model:                        {}".format(arguments.model))
	logging_info("Model name:                   {}".format(arguments.model_name))
	logging_info("Dataset:                      {}".format(arguments.dataset))
	logging_info("Tensorflow verbosity:         {}".format(arguments.tf_verbosity))
	logging_info("Tensorflow minimum log level: {}".format(arguments.tf_min_log_level))

	should_continue = query_yes_no("Continue?", default="yes")
	if not should_continue:
		exit()

	# set verbosity of tensorflow
	tfu_set_logging(arguments.tf_verbosity, min_log_level=arguments.tf_min_log_level)

	# load the graph
	graph_filename = arguments.model_name
	if not graph_filename.endswith(".pb"):
		graph_filename = "{}.pb".format(graph_filename)
	graph_path = get_full_path("models", graph_filename)
	graph = tfu_load_graph(graph_path)

	# load the dataset
	dataset = get_dataset(arguments.dataset)

	# create color palette
	colors = get_distinct_colors(dataset.num_classes)

	# check format of image
	image_format = dataset.get_image_format_of_file(arguments.image_filename)
	if image_format is None:
		raise NotImplementedError("The format of the file '{}' is currently not supported.".format(arguments.image_filename))

	# read image
	with tf.gfile.FastGFile(arguments.image_filename, "rb") as file:
		image_raw_data = file.read()

	# start a new Tensorflow session
	with tf.Session(graph=graph) as session:
		# decode image
		image_placeholder = tf.placeholder(tf.string)
		image_format_placeholder = tf.placeholder(tf.string)
		decode_bmp_fn = lambda: tf.image.decode_bmp(image_placeholder, channels=0)
		decode_jpeg_fn = lambda: tf.image.decode_jpeg(image_placeholder, channels=0, dct_method="INTEGER_FAST")
		decode_png_fn = lambda: tf.image.decode_png(image_placeholder, channels=0)
		decoded_image = tf.case({
			tf.equal(image_format_placeholder, "bmp"): decode_bmp_fn,
			tf.equal(image_format_placeholder, "jpeg"): decode_jpeg_fn,
			tf.equal(image_format_placeholder, "png"): decode_png_fn
		}, exclusive=True)

		image = session.run(decoded_image, feed_dict={
			image_placeholder: image_raw_data,
			image_format_placeholder: image_format
		})

		if arguments.model == "ssd_vgg_300":
			input_shape = (300, 300)
		elif arguments.model == "ssd_vgg_512":
			input_shape = (512, 512)
		else:
			raise RuntimeError("The model '{}' is currently not supported.".format(arguments.model))

		input_image = cv2.resize(image, input_shape, interpolation=cv2.INTER_NEAREST)
		input_image = np.expand_dims(input_image, axis=0)

		# start inference
		input_tensor_name = "{}/input/image:0".format(arguments.model)
		output_classes_tensor_name = "{}/output/classes:0".format(arguments.model)
		output_scores_tensor_name = "{}/output/scores:0".format(arguments.model)
		output_localizations_tensor_name = "{}/output/localizations:0".format(arguments.model)
		result = session.run({
			"classes": graph.get_tensor_by_name(output_classes_tensor_name),
			"scores": graph.get_tensor_by_name(output_scores_tensor_name),
			"localizations": graph.get_tensor_by_name(output_localizations_tensor_name)
		}, feed_dict={
			graph.get_tensor_by_name(input_tensor_name): input_image
		})

		classes = result["classes"]
		scores = result["scores"]
		localizations = result["localizations"]

		# add annotations
		for jj in range(classes[0].shape[0]):
			cl = classes[0][jj]
			if cl == 0:
				continue
			draw_single_box(
				image,
				localizations[0][jj],
				score = scores[0][jj],
				label = dataset.selected_labels[cl - 1],
				color = colors[cl - 1]
			)

		plt.figure()
		plt.imshow(image)
		plt.show()

		logging_info("Successfully finished inference.")
