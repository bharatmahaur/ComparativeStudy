##
## /src/models/layers.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 04/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import numpy as np
import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


def atrous_convolution2d(inputs,
						 num_filters,
						 kernel_size,
						 dilation_rate,
						 padding = "SAME",
						 activation = True,
						 seed = None,
						 name = None):
	"""Build an atrous convolutional layer.

	Arguments:
		inputs: Input tensor.
		num_filters: Number of filters.
		kernel_size: Tuple defining the kernel size.
		dilation_rate: Dilation rate.
		padding: Padding method. Defaults to 'SAME'.
		activation: Flag whether to use ReLU activation. Defaults to True.
		seed: Seed for random operations. Defaults to None.
		name: Name for variable scope of the operation. Defaults to None.

	Returns:
		Output tensor.
	"""
	with tf.variable_scope(name):
		# define weights tensor
		weights = tf.get_variable(
			"weights",
			shape = (kernel_size[0], kernel_size[1], inputs.get_shape()[3], num_filters),
			initializer = tf.contrib.layers.xavier_initializer(seed=seed)
		)

		# define bias tensor
		bias = tf.get_variable(
			"bias",
			shape = (num_filters,),
			initializer = tf.zeros_initializer()
		)

		# build atrous convolutional layer
		output = tf.nn.atrous_conv2d(
			inputs,
			weights,
			dilation_rate,
			padding
		)

		# add bias and activate output
		output = tf.nn.bias_add(output, bias)
		if activation:
			output = tf.nn.relu(output)

		# add l2 loss to collection
		loss = tf.nn.l2_loss(weights)
		tf.add_to_collection("l2_losses", loss)

	return output


def convolution2d(inputs,
				  num_filters,
				  kernel_size,
				  strides,
				  padding = "SAME",
				  activation = True,
				  seed = None,
				  name = None):
	"""Build a convolutional layer.

	Arguments:
		inputs: Input tensor.
		num_filters: Number of filters.
		kernel_size: Tuple defining the kernel size.
		strides: Tuple defining the strides of the layer.
		padding: Padding method. Defaults to 'SAME'.
		activation: Flag whether to use ReLU activation. Defaults to True.
		seed: Seed for random operations. Defaults to None.
		name: Name for variable scope of the operation. Defaults to None.

	Returns:
		Output tensor.
	"""
	with tf.variable_scope(name):
		# define weights tensor
		weights = tf.get_variable(
			"weights",
			shape = (kernel_size[0], kernel_size[1], inputs.get_shape()[3], num_filters),
			initializer = tf.contrib.layers.xavier_initializer(seed=seed)
		)

		# define bias tensor
		bias = tf.get_variable(
			"bias",
			shape = (num_filters,),
			initializer = tf.zeros_initializer()
		)

		# build convolutional layer
		output = tf.nn.conv2d(
			inputs,
			weights,
			[1, strides[0], strides[1], 1],
			padding
		)

		# add bias and activate output
		output = tf.nn.bias_add(output, bias)
		if activation:
			output = tf.nn.relu(output)

		# add l2 loss to collection
		loss = tf.nn.l2_loss(weights)
		tf.add_to_collection("l2_losses", loss)

	return output


def l2_normalization(inputs,
					 initial_scale = None,
					 name = None):
	"""Build a l2 normalization layer.

	Arguments:
		inputs: Input tensor.
		initial_scale: Initial scale of L2 normalization. Defaults to None.
		name: Name for variable scope of the operation. Defaults to None.

	Returns:
		Output tensor.
	"""
	with tf.variable_scope(name):
		# normalize output
		input_shape = inputs.get_shape()
		output = tf.nn.l2_normalize(inputs, axis=-1)

		# scale the output
		if initial_scale is not None:
			scale_value = initial_scale * np.ones(input_shape[-1])
			initializer = tf.constant_initializer(
				value = scale_value,
				dtype = tf.float32
			)
			scale = tf.get_variable(
				name = "scale",
				initializer = initializer,
				shape = scale_value.shape
			)
			output = scale * output

	return output


def max_pooling2d(inputs,
				  kernel_size,
				  strides,
				  padding = "SAME",
				  name = None):
	"""Build a max pooling layer.

	Arguments:
		inputs: Input tensor.
		kernel_size: Tuple defining the kernel size.
		strides: Tuple defining the strides of the layer.
		padding: Padding method. Defaults to 'SAME'.
		name: Name for variable scope of the operation. Defaults to None.
	"""
	return tf.nn.max_pool(
		inputs,
		ksize = [1, kernel_size[0], kernel_size[1], 1],
		strides = [1, strides[0], strides[1], 1],
		padding = padding,
		name = name
	)


def smooth_l1_loss(inputs, name = None):
	"""Build a smooth L1 layer.

	Arguments:
		inputs: Input tensor.
		name: Name for variable scope of the operation. Defaults to None.
	"""
	with tf.variable_scope(name, default_name="smooth_l1_loss"):
		abs = tf.abs(inputs)
		output = tf.where(abs < 1.0, 0.5 * tf.square(inputs), abs - 0.5)
	return output
