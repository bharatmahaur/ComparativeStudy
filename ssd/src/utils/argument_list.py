##
## /src/utils/argument_list.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import argparse
import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.logging import logging_error


class ArgumentList(object):
	"""Handle easy creation of application arguments.

	Attributes:
		parser: Agument parser.
	"""

	def __init__(self, *args, **kwargs):
		"""Initialize the class.

		Arguments:
			args: List of arguments passed to the argument parser.
			kwargs: Dictionary of arguments passed to the argument parser.
		"""
		self.parser = argparse.ArgumentParser(*args, **kwargs)


	@property
	def available_models(self):
		"""List of available model names.
		"""
		return ["ssd_vgg_300", "ssd_vgg_512"]


	@property
	def available_datasets(self):
		"""List of available dataset names.
		"""
		return ["voc2007", "voc2012", "voc2007+2012"]


	@property
	def available_optimizers(self):
		"""List of available optimizer names.
		"""
		return [
			"adadelta",
			"adagrad",
			"adagrad_da",
			"adam",
			"ftrl",
			"gradient_descent",
			"momentum",
			"proximal_adagrad",
			"proximal_gradient_descent",
			"rms_prop"
		]


	@property
	def available_verbosities(self):
		"""List of available TF verbosities.
		"""
		return ["debug", "error", "fatal", "info", "warn"]


	@property
	def available_min_log_levels(self):
		"""List of available TF minimum logging levels.
		"""
		return [0, 1, 2, 3]


	@property
	def required_optimizer_arguments(self):
		"""Dictionary containing required arguments for optimizers.
		"""
		return {
			"momentum": ["momentum"]
		}


	@property
	def supported_image_formats(self):
		"""Tuple of supported image format extensions.
		"""
		return (".bmp", ".dib", ".jfi", ".jfif", ".jif", ".jpe", ".jpeg", ".jpg", ".png")


	def add_run_argument(self, help, required = False):
		"""Add --run argument to the parser.

		Arguments:
			help: Help text for the argument.
			required: Flag whether the argument is required. Defaults to False.
		"""
		self.parser.add_argument(
			"--run",
			type = str,
			required = required,
			help = help
		)


	def add_model_argument(self, help, default = "ssd_vgg_300", required = True):
		"""Add --video-filename argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 'ssd_vgg_300'.
			required: Flag whether the argument is required. Defaults to True.
		"""
		self.parser.add_argument(
			"--model",
			default = default,
			choices = self.available_models,
			required = required,
			help = help
		)


	def add_model_name_argument(self, help, required = True):
		"""Add --model-name argument to the parser.

		Arguments:
			help: Help text for the argument.
			required: Flag whether the argument is required. Defaults to True.
		"""
		self.parser.add_argument(
			"--model-name",
			type = str,
			required = required,
			help = help
		)


	def add_output_nodes_argument(self, help, required = True):
		"""Add --output-nodes argument to the parser.

		Arguments:
			help: Help text for the argument.
			required: Flag whether the argument is required. Defaults to True.
		"""
		self.parser.add_argument(
			"--output-nodes",
			nargs = "+",
			type = str,
			required = required,
			help = help
		)


	def add_image_filename_argument(self, help, required = True):
		"""Add --image-filename argument to the parser.

		Arguments:
			help: Help text for the argument.
			required: Flag whether the argument is required. Defaults to True.
		"""
		self.parser.add_argument(
			"--image-filename",
			type = tf_image_filename_type,
			required = required,
			help = help
		)


	def add_video_filename_argument(self, help, required = True):
		"""Add --video-filename argument to the parser.

		Arguments:
			help: Help text for the argument.
			required: Flag whether the argument is required. Defaults to True.
		"""
		self.parser.add_argument(
			"--video-filename",
			type = str,
			required = required,
			help = help
		)


	def add_dataset_argument(self, help, default = "voc2007", required = True):
		"""Add --dataset argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 'voc2007'.
			required: Flag whether the argument is required. Defaults to True.
		"""
		self.parser.add_argument(
			"--dataset",
			default = default,
			type = str,
			choices = self.available_datasets,
			required = required,
			help = help
		)


	def add_dataset_split_argument(self, help, default = "train", required = True):
		"""Add --dataset-split argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 'train'.
			required: Flag whether the argument is required. Defaults to True.
		"""
		self.parser.add_argument(
			"--dataset-split",
			default = default,
			type = str,
			required = required,
			help = help
		)


	def add_random_seed_argument(self, help, default = 1807241):
		"""Add --random-seed argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 1807241.
		"""
		self.parser.add_argument(
			"--random-seed",
			default = default,
			type = int,
			help = help
		)


	def add_op_random_seed_argument(self, help, default = 1807242):
		"""Add --op-random-seed argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 1807242.
		"""
		self.parser.add_argument(
			"--op-random-seed",
			default = default,
			type = int,
			help = help
		)


	def add_num_parallel_calls_argument(self, help, default = 6):
		"""Add --num-parallel-calls argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 6.
		"""
		self.parser.add_argument(
			"--num-parallel-calls",
			default = default,
			type = positive_int_type,
			help = help
		)


	def add_prefetch_buffer_size_argument(self, help, default = 2):
		"""Add --prefetch-buffer-size argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 2.
		"""
		self.parser.add_argument(
			"--prefetch-buffer-size",
			default = default,
			type = positive_int_type,
			help = help
		)


	def add_shuffle_buffer_size_argument(self, help, default = 10000):
		"""Add --shuffle-buffer-size argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 10000.
		"""
		self.parser.add_argument(
			"--shuffle-buffer-size",
			default = default,
			type = positive_int_type,
			help = help
		)


	def add_batch_size_argument(self, help, default = 32):
		"""Add --batch-size argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 32.
		"""
		self.parser.add_argument(
			"--batch-size",
			default = default,
			type = positive_int_type,
			help = help
		)


	def add_num_steps_argument(self, help, default = []):
		"""Add --num-steps argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to [].
		"""
		self.parser.add_argument(
			"--num-steps",
			nargs = "*",
			default = default,
			type = positive_int_type,
			help = help
		)


	def add_learning_rate_argument(self, help, default = [1e-3]):
		"""Add --learning-rate argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to [1e-3].
		"""
		self.parser.add_argument(
			"--learning-rate",
			nargs = "+",
			default = default,
			type = positive_float_type,
			help = help
		)


	def add_optimizer_argument(self, help, default = "adam"):
		"""Add --optimizer argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 'adam'.
		"""
		self.parser.add_argument(
			"--optimizer",
			default = default,
			type = str,
			choices = self.available_optimizers,
			help = help
		)


	def add_momentum_argument(self, help, default = None):
		"""Add --momentum argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to None.
		"""
		self.parser.add_argument(
			"--momentum",
			default = default,
			type = exclusive_unit_float_type,
			help = help
		)


	def add_input_device_argument(self, help, default = "/cpu:0"):
		"""Add --input-device argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to '/cpu:0'.
		"""
		self.parser.add_argument(
			"--input-device",
			default = default,
			type = str,
			help = help
		)


	def add_inference_device_argument(self, help, default = "/gpu:0"):
		"""Add --inference-device argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to '/gpu:0'.
		"""
		self.parser.add_argument(
			"--inference-device",
			default = default,
			type = str,
			help = help
		)


	def add_optimization_device_argument(self, help, default = "/cpu:0"):
		"""Add --optimization-device argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to '/cpu:0'.
		"""
		self.parser.add_argument(
			"--optimization-device",
			default = default,
			type = str,
			help = help
		)


	def add_checkpoint_interval_argument(self, help, default = 1000):
		"""Add --checkpoint-interval argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 1000.
		"""
		self.parser.add_argument(
			"--checkpoint-interval",
			default = default,
			type = positive_int_type,
			help = help
		)


	def add_step_log_interval_argument(self, help, default = 10):
		"""Add --step-log-interval argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 10.
		"""
		self.parser.add_argument(
			"--step-log-interval",
			default = default,
			type = positive_int_type,
			help = help
		)


	def add_tf_verbosity_argument(self, help, default = "info"):
		"""Add --tf-verbosity argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 'info'.
		"""
		self.parser.add_argument(
			"--tf-verbosity",
			default = default,
			type = str,
			choices = self.available_verbosities,
			help = help
		)


	def add_tf_min_log_level_argument(self, help, default = 3):
		"""Add --tf-min-log-level argument to the parser.

		Arguments:
			help: Help text for the argument.
			default: Default value of the argument. Defaults to 3.
		"""
		self.parser.add_argument(
			"--tf-min-log-level",
			default = default,
			type = int,
			choices = self.available_min_log_levels,
			help = help
		)


	def parse(self):
		"""Parse the arguments of the application.

		Returns:
			Namespace containing the parsed arguments.
		"""
		arguments = self.parser.parse_args()
		arguments_dict = vars(arguments)

		# validate optimizer
		if "optimizer" in arguments_dict:
			required_arguments = self.required_optimizer_arguments.get(arguments.optimizer, [])
			for argument in required_arguments:
				if argument not in arguments_dict or arguments_dict[argument] is None:
					logging_error("For the optimizer '{}' the following arguments are required: {}.".format(
						arguments.optimizer, required_arguments
					))

		# validate number of steps and learning rate
		if "num_steps" in arguments_dict and "learning_rate" in arguments_dict:
			if len(arguments.num_steps) == 0 and len(arguments.learning_rate) != 1:
				logging_error("When no step limit is given, exactly one learning rate must be given.")
			elif len(arguments.num_steps) > 0:
				if len(arguments.num_steps) != len(arguments.learning_rate):
					logging_error("There must be a learning rate (in total {}) for exactly one step block (in total {}).".format(
						len(arguments.learning_rate), len(arguments.num_steps)
					))

		return arguments


def boolean_type(arg):
	"""Check an argument for a boolean type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	if arg.lower() in ("yes", "true", "t", "y", "1"):
		return True
	if arg.lower() in ("no", "false", "f", "n", "0"):
		return False
	raise argparse.ArgumentTypeError("invalid boolean value: '{}'".format(arg))


def exclusive_unit_float_type(arg):
	"""Check an argument for an exclusive unit float type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_float = float(arg)
	if arg_float <= 0.0 or arg_float >= 1.0:
		raise argparse.ArgumentTypeError("invalid exclusive_unit_float_type value: '{}', must be exclusive between 0 and 1".format(arg))
	return arg_float


def positive_int_type(arg):
	"""Check an argument for a positive integer type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_int = int(arg)
	if arg_int <= 0:
		raise argparse.ArgumentTypeError("invalid positive_int_type value: '{}', must be positive".format(arg))
	return arg_int


def positive_float_type(arg):
	"""Check an argument for a positive float type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_float = float(arg)
	if arg_float <= 0.0:
		raise argparse.ArgumentTypeError("invalid positive_float_type value: '{}', must be positive".format(arg))
	return arg_float


def tf_image_filename_type(arg):
	"""Check an argument for a by TF supported image format type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	if not os.path.isfile(arg):
		raise argparse.ArgumentTypeError("invalid tf_image_filename_type value: '{}', must be an existing file.".format(arg))
	arg_lower = arg.lower()
	valid_extensions = (".bmp", ".dib", ".jfi", ".jfif", ".jif", ".jpe", ".jpeg", ".jpg", ".png")
	if not arg_lower.endswith(valid_extensions):
		raise argparse.ArgumentTypeError("invalid tf_image_filename_type value: '{}', must be one of the formats {}.".format(
			arg, valid_extensions
		))
	return arg
