##
## /src/train.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 04/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import atexit
import numpy as np
import os
import shutil
import sys
import tensorflow as tf
from datetime import datetime

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from data import DataProvider
from data.preprocessors import BBoxPreprocessor, DefaultPreprocessor, ImagePreprocessor
from datasets.common.utils import get_dataset
from models.ssd.common.utils import get_model
from utils import ArgumentList, AveragePrecision, Run, Summary
from utils.common.logging import logging_error, logging_info, logging_train
from utils.common.terminal import query_yes_no
from utils.tfu import tfu_get_optimizer, tfu_get_uninitialized_variables, tfu_set_logging


def exit_handler(run):
	"""Function that is called before the script exits.

	Arguments:
		run: The run that needs exit handling.
	"""
	force_drop_run = True

	# remove run if no checkpoint was saved
	for file in os.listdir(run.checkpoints_path):
		if file.startswith("model") or file == "checkpoint":
			force_drop_run = False
			break
	if force_drop_run and os.path.exists(run.base_path):
		shutil.rmtree(run.base_path)

	# skip if run does not exist
	if not os.path.exists(run.base_path):
		return

	# ask user to keep the run
	should_keep_run = query_yes_no(
		"Should the run '{}' be kept?".format(os.path.basename(run.base_path)),
		default="yes"
	)
	if not should_keep_run:
		shutil.rmtree(run.base_path)


if __name__ == "__main__":

	# parse arguments
	argument_list = ArgumentList(
		description = "Train a model for object detection."
	)
	argument_list.add_model_argument("The model to use for training.", default="ssd_vgg_300")
	argument_list.add_dataset_argument("The dataset to use for training.", default="voc2007")
	argument_list.add_dataset_split_argument("The dataset split to use for training.", default="train", required=False)
	argument_list.add_random_seed_argument("The global random seed used for determinism.", default=1807241)
	argument_list.add_op_random_seed_argument("The operation random seed used for determinism.", default=1807242)
	argument_list.add_num_parallel_calls_argument("Number of parallel calls for preprocessing the data.", default=6)
	argument_list.add_prefetch_buffer_size_argument("Buffer size for prefetching the data.", default=2)
	argument_list.add_shuffle_buffer_size_argument("Buffer size for shuffling the data.", default=10000)
	argument_list.add_batch_size_argument("Batch size for training.", default=32)
	argument_list.add_num_steps_argument("Number of steps for training.", default=[])
	argument_list.add_learning_rate_argument("Learning rate of the optimizer. Must be given for each block of steps.", default=[1e-3])
	argument_list.add_optimizer_argument("Optimizer for training.", default="momentum")
	argument_list.add_momentum_argument("Mommentum of the optimizer. Only used with the Momentum optimizer.", default=0.9)
	argument_list.add_input_device_argument("Device for processing inputs.", default="/cpu:0")
	argument_list.add_inference_device_argument("Device for inference.", default="/gpu:0")
	argument_list.add_optimization_device_argument("Device for optimization.", default="/cpu:0")
	argument_list.add_checkpoint_interval_argument("Interval how often the model should be saved [steps].", default=1000)
	argument_list.add_step_log_interval_argument("Interval how often step information should be displayed [steps].", default=10)
	argument_list.add_tf_verbosity_argument("Tensorflow verbosity.", default="info")
	argument_list.add_tf_min_log_level_argument("Tensorflow minimum log level.", default=3)
	arguments = argument_list.parse()

	# get dataset
	dataset = get_dataset(arguments.dataset)

	# initialize new run
	run = Run(run_id=None)
	run.open()
	run.set_config_value(arguments.model, "model", "name")
	run.set_config_value(arguments.dataset, "model", "dataset")
	run.set_config_value(arguments.dataset_split, "model", "dataset_split")
	run.set_config_value(dataset.num_classes + 1, "model", "num_classes")
	run.set_config_value(arguments.random_seed, "training", "random_seed")
	run.set_config_value(arguments.op_random_seed, "training", "op_random_seed")
	run.set_config_value(arguments.num_parallel_calls, "training", "num_parallel_calls")
	run.set_config_value(arguments.prefetch_buffer_size, "training", "prefetch_buffer_size")
	run.set_config_value(arguments.shuffle_buffer_size, "training", "shuffle_buffer_size")
	run.set_config_value(arguments.batch_size, "training", "batch_size")
	run.set_config_value(arguments.num_steps, "training", "num_steps")
	run.set_config_value(arguments.learning_rate, "training", "learning_rate")
	run.set_config_value(arguments.optimizer, "training", "optimizer")
	if arguments.optimizer == "momentum":
		run.set_config_value(arguments.momentum, "training", "momentum")
	run.set_config_value(arguments.input_device, "training", "input_device")
	run.set_config_value(arguments.inference_device, "training", "inference_device")
	run.set_config_value(arguments.optimization_device, "training", "optimization_device")
	run.set_config_value(arguments.checkpoint_interval, "training", "checkpoint_interval")
	run.set_config_value(arguments.step_log_interval, "training", "step_log_interval")
	run.set_config_value(datetime.utcnow().isoformat(), "misc", "creation_date")
	run.set_config_value(np.__version__, "misc", "np_version")
	run.set_config_value(tf.__version__, "misc", "tf_version")
	run.set_config_value(arguments.tf_verbosity, "misc", "tf_verbosity")
	run.set_config_value(arguments.tf_min_log_level, "misc", "tf_min_log_level")
	run.save_config()

	# print some information
	logging_info("Initialized run '{}'.".format(run.id))
	logging_info("Model:                        {}".format(arguments.model))
	logging_info("Dataset:                      {} {}".format(arguments.dataset, arguments.dataset_split))
	logging_info("Number of classes:            {}".format(dataset.num_classes + 1))
	logging_info("Global random seed:           {}".format(arguments.random_seed))
	logging_info("Operation random seed:        {}".format(arguments.op_random_seed))
	logging_info("Preprocessing parallel calls: {}".format(arguments.num_parallel_calls))
	logging_info("Prefetch buffer size:         {}".format(arguments.prefetch_buffer_size))
	logging_info("Shuffle buffer size:          {}".format(arguments.shuffle_buffer_size))
	logging_info("Batch size:                   {}".format(arguments.batch_size))
	logging_info("Number of steps:              {}".format(arguments.num_steps))
	logging_info("Learning rate:                {}".format(arguments.learning_rate))
	logging_info("Optimizer:                    {}".format(arguments.optimizer))
	if arguments.optimizer == "momentum":
		logging_info("Momentum:                     {}".format(arguments.momentum))
	logging_info("Input device:                 {}".format(arguments.input_device))
	logging_info("Inference device:             {}".format(arguments.inference_device))
	logging_info("Optimization device:          {}".format(arguments.optimization_device))
	logging_info("Checkpoint interval:          {}".format(arguments.checkpoint_interval))
	logging_info("Step log interval:            {}".format(arguments.step_log_interval))
	logging_info("NumPy version:                {}".format(np.__version__))
	logging_info("Tensorflow version:           {}".format(tf.__version__))
	logging_info("Tensorflow verbosity:         {}".format(arguments.tf_verbosity))
	logging_info("Tensorflow minimum log level: {}".format(arguments.tf_min_log_level))

	# register exit handler
	atexit.register(exit_handler, run)

	should_continue = query_yes_no("Continue?", default="yes")
	if not should_continue:
		exit()

	# print some information about tensorboard
	logging_info("Start Tensorboard for visualization from the command line: 'tensorboard --logdir training/'.")
	logging_info("Tensorboard is available in your browser: http://127.0.0.1:6006.")

	# set verbosity of tensorflow
	tfu_set_logging(arguments.tf_verbosity, min_log_level=arguments.tf_min_log_level)

	# start a new Tensorflow session
	with tf.Session() as session:
		# set random seed
		tf.set_random_seed(arguments.random_seed)

		# init model
		model = get_model(
			arguments.model,
			session,
			op_seed = arguments.op_random_seed,
			num_classes = dataset.num_classes + 1
		)

		with tf.device(arguments.input_device):
			# setup data provider
			dataset_shuffle_seed = tf.placeholder(tf.int64, (), name="dataset_shuffle_seed")
			data_provider = DataProvider(arguments.dataset)
			data_provider.init_dataset(
				arguments.dataset_split,
				batch_size = arguments.batch_size,
				num_parallel_calls = arguments.num_parallel_calls,
				prefetch_buffer_size = arguments.prefetch_buffer_size,
				shuffle_buffer_size = arguments.shuffle_buffer_size,
				shuffle_random_seed = dataset_shuffle_seed,
				preprocessor = [
					DefaultPreprocessor(),
					ImagePreprocessor(model.hyperparams["image_shape"], op_seed=arguments.op_random_seed, data_augmentation=True),
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
			training = True,
			inference_device = arguments.inference_device,
			optimization_device = arguments.optimization_device
		)
		restore_ops = model.restore_vgg_16()

		# setup counter variables
		epoch = tf.Variable(1, name="epoch", trainable=False, dtype=tf.int64)
		global_step = tf.Variable(1, name="global_step", trainable=False, dtype=tf.int64)

		# setup optimizer
		learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
		optimizer_kwargs = {}
		if arguments.optimizer == "momentum":
			optimizer_kwargs["momentum"] = arguments.momentum

		optimizer = tfu_get_optimizer(arguments.optimizer, learning_rate, **optimizer_kwargs)
		if optimizer is None:
			logging_error("The optimizer '{}' is not supported.".format(arguments.optimizer))

		epoch_op = tf.assign(epoch, tf.add(epoch, 1))
		train_op = optimizer.minimize(model.losses["total"], global_step=global_step)

		# initialize variables
		uninitialized_variables = tfu_get_uninitialized_variables(session)
		if len(uninitialized_variables) > 0:
			session.run(tf.variables_initializer(uninitialized_variables))
		session.run(restore_ops)

		# initialize saver
		saver = tf.train.Saver(max_to_keep=100)

		# setup summaries
		summary_writer = tf.summary.FileWriter(run.base_path, session.graph)

		summary = Summary(session, summary_writer)
		summary.init_losses()
		summary.init_average_precisions(dataset.selected_labels)
		summary.init_mean_average_precision()
		summary.merge()

		# save graph.pbtxt
		tf.train.write_graph(session.graph_def, run.base_path, "graph.pbtxt")

		# setup some other stuff
		average_precision = AveragePrecision(matching_threshold=0.5)

		# initialize training loop
		epoch_value = session.run(epoch)
		global_step_value = session.run(global_step)
		logging_info("Training from epoch {}, step {}.".format(epoch_value, global_step_value))

		# get number of steps
		num_steps = arguments.num_steps
		learning_rates = arguments.learning_rate
		max_steps = np.sum(num_steps)

		# loop
		while True:
			epoch_value = session.run(epoch)

			# training loop
			session.run(data_provider.get_initializer(arguments.dataset_split), feed_dict={
				dataset_shuffle_seed: 1
			})
			while True:
				# get current learning rate
				temp = 0
				learning_rate_value = learning_rates[0]
				for ii in range(len(num_steps)):
					temp += num_steps[ii]
					if global_step_value < temp:
						learning_rate_value = learning_rates[ii]
						break

				# run a step
				try:
					batch_data = session.run(batch)
					result = session.run({
						"train": train_op,
						"losses": model.losses,
						"output": model.output,
						"global_step_value": global_step,
						"learning_rate_value": learning_rate
					}, feed_dict={
						model.image_input: batch_data["image"],
						model.groundtruth_classes: batch_data["image/object/encoding/bbox/class"],
						model.groundtruth_scores: batch_data["image/object/encoding/bbox/score"],
						model.groundtruth_localizations: batch_data["image/object/encoding/bbox"],
						learning_rate: learning_rate_value
					})
				except tf.errors.OutOfRangeError:
					break

				losses = result["losses"]
				global_step_value = result["global_step_value"]
				learning_rate_value = result["learning_rate_value"]

				# add predictions for average precision calculation
				if epoch_value > 1:
					average_precision.add_predictions(
						batch_data["image/object/bbox/label"],
						batch_data["image/object/bbox"],
						result["output"]["classes"],
						result["output"]["scores"],
						result["output"]["localizations"]
					)

				# add losses to summary
				summary.set_loss("confidence", losses["confidence"])
				summary.set_loss("localization", losses["localization"])
				summary.set_loss("l2", losses["l2"])
				summary.set_loss("total", losses["total"])
				summary.write_loss(global_step_value)

				# print some information
				if global_step_value % arguments.step_log_interval == 0:
					logging_train("Step {}: epoch {}, learning rate {:7.5f}, loss {:7.5f}".format(
						global_step_value,
						epoch_value,
						learning_rate_value,
						losses["total"]
					))

				# save checkpoint
				if global_step_value % arguments.checkpoint_interval == 0:
					checkpoint_path = run.checkpoints_file_path
					logging_train("Saving to '{}-{}'.".format(checkpoint_path, global_step_value))
					saver.save(session, checkpoint_path, global_step=global_step_value)

				# stop training
				if max_steps > 0 and global_step_value >= max_steps:
					logging_train("Trained for a total of {} steps. Finished.".format(global_step_value))
					break

			# compute average precision values
			if not average_precision.is_clean:
				logging_train("Matching predictions with groundtruth values.")

				cl_precision, cl_recall = average_precision.precision_recall()
				cl_ap = average_precision.average_precision(cl_precision, cl_recall)
				if len(cl_ap) > 0:
					for cl in sorted(cl_ap):
						logging_train("AP[cl={}] = {}".format(cl, cl_ap[cl]))
					mAP = np.mean(list(cl_ap.values()))
					logging_train("mAP = {}".format(mAP))
				else:
					logging_train("No classes were predicted.")

				# add average precision values to summary
				for ii, label in enumerate(dataset.selected_labels):
					value = cl_ap.get(ii + 1, 0.0)
					summary.set_average_precision(label, value)

				# add mean average precision to summary
				summary.set_mean_average_precision(0.0 if len(cl_ap) == 0 else mAP)

				average_precision.clear()

				# write train summaries
				summary.write_average_precision(global_step_value)
				summary.write_mean_average_precision(global_step_value)

			# stop training
			if max_steps > 0 and global_step_value >= max_steps:
				break

			# increase epoch value
			session.run(epoch_op)
