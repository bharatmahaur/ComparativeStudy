##
## /src/models/eval.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 08/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import numpy as np
import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from data import DataProvider
from data.preprocessors import BBoxPreprocessor, DefaultPreprocessor, ImagePreprocessor
from datasets.common.utils import get_dataset
from models.ssd.common.utils import get_model
from utils import ArgumentList, AveragePrecision, Run
from utils.common.logging import logging_error, logging_info, logging_eval
from utils.common.terminal import query_yes_no
from utils.tfu import tfu_get_uninitialized_variables, tfu_set_logging


if __name__ == "__main__":

	# parse arguments
	argument_list = ArgumentList(
		description = "Evaluate model from a run and compute mAP."
	)
	argument_list.add_run_argument("The run from which to evaluate the model.", required=True)
	argument_list.add_dataset_argument("The dataset to use for evaluation.", default="voc2007")
	argument_list.add_dataset_split_argument("The dataset split to use for evaluation.", default="test")
	argument_list.add_num_parallel_calls_argument("Number of parallel calls for preprocessing the data.", default=6)
	argument_list.add_prefetch_buffer_size_argument("Buffer size for prefetching the data.", default=2)
	argument_list.add_batch_size_argument("Batch size for evaluation.", default=32)
	argument_list.add_input_device_argument("Device for processing inputs.", default="/cpu:0")
	argument_list.add_inference_device_argument("Device for inference.", default="/gpu:0")
	argument_list.add_optimization_device_argument("Device for optimization.", default="/cpu:0")
	argument_list.add_tf_verbosity_argument("Tensorflow verbosity.", default="info")
	argument_list.add_tf_min_log_level_argument("Tensorflow minimum log level.", default=3)
	arguments = argument_list.parse()

	# load run
	run = Run(run_id=arguments.run)
	if not run.open():
		logging_error("There is no run '{}'.".format(arguments.run))

	# print some information
	logging_info("Load run '{}'.".format(arguments.run))
	logging_info("Model:                        {}".format(run.get_config_value("model", "name")))
	logging_info("Dataset:                      {} {}".format(arguments.dataset, arguments.dataset_split))
	logging_info("Preprocessing parallel calls: {}".format(arguments.num_parallel_calls))
	logging_info("Prefetch buffer size:         {}".format(arguments.prefetch_buffer_size))
	logging_info("Batch size:                   {}".format(arguments.batch_size))
	logging_info("Input device:                 {}".format(arguments.input_device))
	logging_info("Inference device:             {}".format(arguments.inference_device))
	logging_info("Optimization device:          {}".format(arguments.optimization_device))
	logging_info("Tensorflow verbosity:         {}".format(arguments.tf_verbosity))
	logging_info("Tensorflow minimum log level: {}".format(arguments.tf_min_log_level))

	should_continue = query_yes_no("Continue?", default="yes")
	if not should_continue:
		exit()

	# get dataset
	dataset = get_dataset(arguments.dataset)

	# set verbosity of tensorflow
	tfu_set_logging(arguments.tf_verbosity, min_log_level=arguments.tf_min_log_level)

	# start a new Tensorflow session
	with tf.Session() as session:
		# get and set random seeds
		random_seed = run.get_config_value("model", "random_seed")
		op_random_seed = run.get_config_value("model", "op_random_seed")
		tf.set_random_seed(random_seed)

		# init model
		model = get_model(
			run.get_config_value("model", "name"),
			session,
			op_seed = op_random_seed,
			num_classes = dataset.num_classes + 1
		)

		with tf.device(arguments.input_device):
			# setup data provider
			data_provider = DataProvider(arguments.dataset)
			data_provider.init_dataset(
				arguments.dataset_split,
				batch_size = arguments.batch_size,
				num_parallel_calls = arguments.num_parallel_calls,
				prefetch_buffer_size = arguments.prefetch_buffer_size,
				preprocessor = [
					DefaultPreprocessor(),
					ImagePreprocessor(model.hyperparams["image_shape"], data_augmentation=False),
					BBoxPreprocessor(
						model.get_default_anchor_boxes(),
						model.hyperparams["prior_scaling"],
						model.hyperparams["matching_threshold"]
					)
				],
				features = [
					"image",
					"image/object/bbox",
					"image/object/bbox/label",
					"image/object/encoding/bbox",
					"image/object/encoding/bbox/class",
					"image/object/encoding/bbox/score"
				],
				padded_shapes = {
					"image": model.hyperparams["tf_image_shape"],
					"image/object/bbox": [None, 4],
					"image/object/bbox/label": [None],
					"image/object/encoding/bbox": [None, 4],
					"image/object/encoding/bbox/class": [None],
					"image/object/encoding/bbox/score": [None]
				},
				name = "data_provider"
			)

			# get batch
			batch = data_provider.get(arguments.dataset_split)

		# build model
		model.build_from_scratch(
			weight_decay = 5e-5,
			training = False,
			inference_device = arguments.inference_device,
			optimization_device = arguments.optimization_device
		)

		# initialize variables
		uninitialized_variables = tfu_get_uninitialized_variables(session)
		if len(uninitialized_variables) > 0:
			session.run(tf.variables_initializer(uninitialized_variables))

		# restore model
		saver = tf.train.Saver()
		latest_checkpoint = tf.train.latest_checkpoint(run.checkpoints_path)
		saver.restore(session, latest_checkpoint)

		# setup some other stuff
		average_precision = AveragePrecision(matching_threshold=0.5)

		# evaluation loop
		batch_index = 0
		session.run(data_provider.get_initializer(arguments.dataset_split))
		while True:
			# run a step
			try:
				batch_data = session.run(batch)
				result = session.run({
					"output": model.output
				}, feed_dict={
					model.image_input: batch_data["image"],
					model.groundtruth_classes: batch_data["image/object/encoding/bbox/class"],
					model.groundtruth_scores: batch_data["image/object/encoding/bbox/score"],
					model.groundtruth_localizations: batch_data["image/object/encoding/bbox"]
				})
			except tf.errors.OutOfRangeError:
				break

			# add predictions for average precision calculation
			average_precision.add_predictions(
				batch_data["image/object/bbox/label"],
				batch_data["image/object/bbox"],
				result["output"]["classes"],
				result["output"]["scores"],
				result["output"]["localizations"]
			)

			logging_eval("Processed batch #{}.".format(batch_index + 1))
			batch_index += 1

		# compute average precision values
		logging_eval("Matching predictions with groundtruth values.")
		cl_precision, cl_recall = average_precision.precision_recall()
		cl_ap = average_precision.average_precision(cl_precision, cl_recall)
		for cl in sorted(cl_ap):
			logging_eval("AP[cl={}] = {}".format(cl, cl_ap[cl]))
		mAP = np.mean(list(cl_ap.values()))
		logging_eval("mAP = {}".format(mAP))

		logging_eval("Successfully finished evaluation.")
