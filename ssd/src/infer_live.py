##
## /src/models/infer_live.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 14/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time
from matplotlib.animation import FuncAnimation

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from datasets.common.utils import get_dataset
from utils import ArgumentList, Run
from utils.common.files import get_full_path
from utils.common.logging import logging_error, logging_info
from utils.common.static import static_vars
from utils.common.terminal import query_yes_no
from utils.tfu import tfu_load_graph, tfu_set_logging
from utils.visualization import draw_fps, draw_single_box, get_distinct_colors


def grab_single_frame(capture):
	"""Grab a single frame from the capture device.

	Arguments:
		capture: OpenCV capture device.

	Returns:
		Single frame.
	"""
	result, frame = capture.read()
	if not result:
		return None

	# convert channel ordering
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	return frame


@static_vars(fps_weight=0.7, fps=None)
def update(ii, session, io, labels, colors, capture, image, should_flip):
	"""Predict boxes and update the frame.

	Arguments:
		ii: Frame index.
		session: TF session.
		io: Dictionary containing input / output tensors.
		labels: List containing labels.
		colors: List containing colors.
		capture: OpenCV capture device.
		image: Matplotlib image.
		should_flip: Flag whether to flip the image from the capture device.
	"""
	start_time = time.time()

	# grab single frame
	frame = grab_single_frame(capture)
	if should_flip:
		frame = cv2.flip(frame, 1)

	# resize frame for inference
	frame_model = cv2.resize(frame, io["input_shape"], interpolation=cv2.INTER_NEAREST)
	frame_model = np.expand_dims(frame_model, axis=0)

	# predict boxes
	result = session.run({
		"classes": io["output"]["classes"],
		"scores": io["output"]["scores"],
		"localizations": io["output"]["localizations"]
	}, feed_dict={
		io["input"]: frame_model
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
			frame,
			localizations[0][jj],
			score = scores[0][jj],
			label = labels[cl - 1],
			color = colors[cl - 1]
		)

	# compute and draw fps
	end_time = time.time()
	if update.fps is None:
		update.fps = 1.0 / (end_time - start_time)
	else:
		update.fps = (1.0 - update.fps_weight) / (end_time - start_time) + update.fps_weight * update.fps
	draw_fps(frame, update.fps)

	# update image
	image.set_data(frame)


def handle_key_press_event(event):
	"""Handle any key press events in the Matplotlib window.
	"""
	if event.key == "q":
		plt.close(event.canvas.figure)


if __name__ == "__main__":

	# parse arguments
	argument_list = ArgumentList(
		description = "Apply an exported model on live data like a video or a webcam."
	)
	argument_list.add_video_filename_argument("The input video filename.", required=False)
	argument_list.add_model_argument("The model used for training.", default=None, required=True)
	argument_list.add_model_name_argument("The exported model name.", required=True)
	argument_list.add_dataset_argument("The dataset used for training.", default=None)
	argument_list.add_tf_verbosity_argument("Tensorflow verbosity.", default="info")
	argument_list.add_tf_min_log_level_argument("Tensorflow minimum log level.", default=3)
	arguments = argument_list.parse()

	# print some information
	if arguments.video_filename is not None:
		logging_info("Video filename:               {}".format(arguments.video_filename))
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

	# get source
	capture_source = 0
	if arguments.video_filename is not None:
		capture_source = arguments.video_filename
	is_webcam = capture_source == 0

	# start a new Tensorflow session
	with tf.Session(graph=graph) as session:
		# capture the video
		capture = cv2.VideoCapture(capture_source)
		if not capture.isOpened():
			logging_error("There was a problem opening the capture device.")
		capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

		# hide the toolbar
		mpl.rcParams["toolbar"] = "None"

		# fit window to image
		plt.figure()
		plt.axis("off")
		image = plt.imshow(grab_single_frame(capture))
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

		# fetch input and output tensors
		if arguments.model == "ssd_vgg_300":
			input_shape = (300, 300)
		elif arguments.model == "ssd_vgg_512":
			input_shape = (512, 512)
		else:
			raise RuntimeError("The model '{}' is currently not supported.".format(arguments.model))
		input_tensor_name = "{}/input/image:0".format(arguments.model)
		output_classes_tensor_name = "{}/output/classes:0".format(arguments.model)
		output_scores_tensor_name = "{}/output/scores:0".format(arguments.model)
		output_localizations_tensor_name = "{}/output/localizations:0".format(arguments.model)
		io = {
			"input_shape": input_shape,
			"input": graph.get_tensor_by_name(input_tensor_name),
			"output": {
				"classes": graph.get_tensor_by_name(output_classes_tensor_name),
				"scores": graph.get_tensor_by_name(output_scores_tensor_name),
				"localizations": graph.get_tensor_by_name(output_localizations_tensor_name)
			}
		}

		# setup animation
		animation = FuncAnimation(
			plt.gcf(),
			update,
			fargs = (session, io, dataset.selected_labels, colors, capture, image, is_webcam),
			interval = 0
		)
		plt.gcf().canvas.mpl_connect("key_press_event", handle_key_press_event)

		# show inference
		plt.show()

		# release capture of webcam
		capture.release()
