##
## /src/freeze.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 14/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import numpy as np
import os
import shutil
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils import ArgumentList, Run
from utils.common.files import get_full_path
from utils.common.logging import logging_error, logging_info
from utils.common.terminal import query_yes_no
from utils.tfu import tfu_set_logging


if __name__ == "__main__":

	# parse arguments
	argument_list = ArgumentList(
		description = "Export the model from a run for inference."
	)
	argument_list.add_run_argument("The run from which to export the model.", required=True)
	argument_list.add_model_name_argument("The output model name.", required=True)
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
	logging_info("Model name:                   {}".format(arguments.model_name))
	logging_info("Tensorflow verbosity:         {}".format(arguments.tf_verbosity))
	logging_info("Tensorflow minimum log level: {}".format(arguments.tf_min_log_level))

	should_continue = query_yes_no("Continue?", default="yes")
	if not should_continue:
		exit()

	# set verbosity of tensorflow
	tfu_set_logging(arguments.tf_verbosity, min_log_level=arguments.tf_min_log_level)

	# define output nodes
	model_name = run.get_config_value("model", "name")
	output_nodes = [
		"{}/output/classes".format(model_name),
		"{}/output/scores".format(model_name),
		"{}/output/localizations".format(model_name)
	]

	# start a new Tensorflow session
	with tf.Session() as session:
		# import meta graph
		latest_checkpoint = tf.train.latest_checkpoint(run.checkpoints_path)
		meta_path = "{}.meta".format(latest_checkpoint)
		saver = tf.train.import_meta_graph(meta_path, clear_devices=True)

		# restore weights
		saver.restore(session, latest_checkpoint)

		# export variables to constants
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			session,
			tf.get_default_graph().as_graph_def(),
			output_nodes
		)

		# write frozen graph
		output_filename = arguments.model_name
		if not output_filename.endswith(".pb"):
			output_filename = "{}.pb".format(output_filename)

		frozen_graph_path = get_full_path("models", output_filename)
		with tf.gfile.FastGFile(frozen_graph_path, "wb") as file:
			file.write(output_graph_def.SerializeToString())

		logging_info("Successfully exported model for inference.")
