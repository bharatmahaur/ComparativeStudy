##
## /src/utils/tfu.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


def tfu_load_graph(filename, name = ""):
	"""Load graph definition from a file.

	Arguments:
		fiilename: Path to the graph to load.
		name: Prefix name for the graph operations.

	Returns:
		Loaded TF graph from file.
	"""
	with tf.gfile.FastGFile(filename, "rb") as file:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(file.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name=name)

	return graph


def tfu_get_uninitialized_variables(session):
	"""Get variables that are not initialized yet.

	Arguments:
		session: TF session.

	Returns:
		List of uninitialized variables.
	"""
	global_variables = tf.global_variables()
	initialized_mask = session.run([
		tf.is_variable_initialized(variable) for variable in global_variables
	])
	uninitialized_variables = [
		variable for (variable, initialized) in zip(global_variables, initialized_mask) if not initialized
	]
	return uninitialized_variables


def tfu_get_optimizer(optimizer_name, *args, **kwargs):
	"""Get a TF optimizer by name.

	Arguments:
		optimizer_name: Name of the optimizer.
		args: List of arguments passed to the initializer of the optimizer.
		kwargs: Dictionary of arguments passed to the initializer of the optimizer.

	Returns:
		Initialized optimizer.
	"""
	optimizer_name = optimizer_name.lower()
	if optimizer_name == "adadelta":
		optimizer = tf.train.AdadeltaOptimizer
	elif optimizer_name == "adagrad":
		optimizer = tf.train.AdagradOptimizer
	elif optimizer_name == "adagrad_da":
		optimizer = tf.train.AdagradDAOptimizer
	elif optimizer_name == "adam":
		optimizer = tf.train.AdamOptimizer
	elif optimizer_name == "ftrl":
		optimizer = tf.train.FtrlOptimizer
	elif optimizer_name == "gradient_descent":
		optimizer = tf.train.GradientDescentOptimizer
	elif optimizer_name == "momentum":
		optimizer = tf.train.MomentumOptimizer
	elif optimizer_name == "proximal_adagrad":
		optimizer = tf.train.ProximalAdagradOptimizer
	elif optimizer_name == "proximal_gradient_descent":
		optimizer = tf.train.ProximalGradientDescentOptimizer
	elif optimizer_name == "rms_prop":
		optimizer = tf.train.RMSPropOptimizer
	else:
		return None

	return optimizer(*args, **kwargs)


def tfu_set_logging(logging, min_log_level = 0):
	"""Set logging verbosity of TensorFlow.

	Arguments:
		logging: Verbosity of TensorFlow.
		min_log_level: Minimum logging level.
	"""
	logging = logging.lower()
	if logging == "debug":
		tf.logging.set_verbosity(tf.logging.DEBUG)
	elif logging == "error":
		tf.logging.set_verbosity(tf.logging.ERROR)
	elif logging == "fatal":
		tf.logging.set_verbosity(tf.logging.FATAL)
	elif logging == "info":
		tf.logging.set_verbosity(tf.logging.INFO)
	elif logging == "warn":
		tf.logging.set_verbosity(tf.logging.WARN)
	else:
		tf.logging.set_verbosity(tf.logging.INFO)

	os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(min_log_level)
