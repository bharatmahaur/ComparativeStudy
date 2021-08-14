##
## /src/models/ssd/ssd_vgg_base.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 23/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import math
import numpy as np
import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from models.custom_layers import atrous_convolution2d, convolution2d, l2_normalization, max_pooling2d, smooth_l1_loss
from models.ssd.common import np_ssd_default_anchor_boxes, tf_ssd_decode_boxes, tf_ssd_select_boxes, tf_ssd_suppress_overlaps
from utils.common.files import get_full_path


class SSDVGGBase(object):
	"""Handle a base SSD network based on a VGG 16 network.

	Attributes:
		session: Current TF session.
		op_seed: Seed for random operations.
		hyperparams: Dictionary containing all hyperparameters of the network.
		feature_layers: Dictionary containing all feature layers.
		losses: Dictionary containing all losses.
		inputs: Placeholder for input image.
		groundtruth_classes: Placeholder for encoded groundtruth classes.
		groundtruth_scores: Placeholder for encoded groundtruth scores.
		groundtruth_localizations: Placeholder for encoded groundtruth boxes.
		ssd_inputs: Input tensor for extra SSD feature layers.
		logits: Output tensor defining predicted logits.
		predictions: Output tensor defining predictions of classes.
		localizations: Output tensor defining predicted localizations.
		__default_anchor_boxes: Default anchor boxes of the network.
		__model_built: Flag whether the model was already built.
	"""

	default_hyperparams = {}
	network_name = None


	def __init__(self, session, op_seed = None, **kwargs):
		"""Initialize the class.

		Arguments:
			session: Current TF session.
			op_seed: Seed for random operations. Defaults to None.
			kwargs: Dictionary containing custom hyperparameters.
		"""
		self.session = session
		self.op_seed = op_seed

		# set hyperparameters of network
		self.hyperparams = kwargs
		for name, value in self.default_hyperparams.items():
			if name in self.hyperparams:
				continue
			self.hyperparams[name] = value

		self.feature_layers = {}
		self.losses = {}

		self.image_input = None
		self.groundtruth_classes = None
		self.groundtruth_scores = None
		self.groundtruth_localizations = None

		self.ssd_inputs = None

		self.logits = None
		self.predictions = None
		self.localizations = None

		self.__default_anchor_boxes = None
		self.__model_built = False


	def build_from_scratch(self,
						   weight_decay = 0.0005,
						   training = True,
						   inference_device = "/gpu:0",
						   optimization_device = "/cpu:0"):
		"""Build the network with all needed layers.

		Arguments:
			weight_decay: Weight of the L2 losses. Defaults to 1e-4.
			training: Flag whether to build the model in train mode. Defaults to True.
			inference_device: Device used for inference. Defaults to '/gpu:0'.
			optimization_device: Device used for optimization. Defaults to '/cpu:0'.
		"""
		# check whether model was already built
		if self.__model_built:
			raise Exception("This model is already built.")

		with tf.variable_scope(self.network_name, default_name="ssd_vgg"):
			# initialize placeholders for inputs
			with tf.variable_scope("input"):
				self.image_input = tf.placeholder(
					tf.float32,
					name = "image",
					shape = (None, self.hyperparams["image_shape"][0], self.hyperparams["image_shape"][1], 3)
				)
				self.groundtruth_classes = tf.placeholder(
					tf.int64,
					shape = (None, self.hyperparams["num_anchors"]),
					name = "classes"
				)
				self.groundtruth_scores = tf.placeholder(
					tf.float32,
					shape = (None, self.hyperparams["num_anchors"]),
					name = "scores"
				)
				self.groundtruth_localizations = tf.placeholder(
					tf.float32,
					shape = (None, self.hyperparams["num_anchors"], 4),
					name = "localizations"
				)

			# build inference layers
			with tf.device(inference_device):
				self.construct_vgg_layers(training=training)
				self.construct_extra_feature_layers(training=training)
				self.construct_multibox_layers(training=training)

			# build optimization layers
			with tf.device(optimization_device):
				self.construct_loss(weight_decay=weight_decay)
				self.construct_post_processing()

		self.__model_built = True


	def get_default_anchor_boxes(self):
		"""Compute default anchor boxes for the network.

		Returns:
			Array of shape (num_aboxes, 4) containing the default anchor boxes.
		"""
		if self.__default_anchor_boxes is None:
			self.__default_anchor_boxes = np_ssd_default_anchor_boxes(
				self.hyperparams["image_shape"],
				self.hyperparams["feature_shapes"],
				self.hyperparams["anchor_ratios"],
				self.hyperparams["anchor_steps"],
				self.hyperparams["anchor_scales"],
				self.hyperparams["anchor_extra_scale"],
				self.hyperparams["anchor_offset"]
			)
		return self.__default_anchor_boxes


	def restore_vgg_16(self, name=None):
		"""Restore the weights and biases from pre-trained VGG 16 network.

		Arguments:
			name: Name for variable scope of the operation. Defaults to None.

		Returns:
			List of operations for restoring the weights.
		"""
		with tf.variable_scope(name, default_name="vgg_16"):
			# initialize checkpoint reader
			model_path = get_full_path("models", "vgg_16_imagenet","vgg_16.ckpt")
			reader = tf.train.NewCheckpointReader(model_path)

			init_biases = {}
			init_weights = {}

			# load biases of all layers
			init_biases["conv1_1"] = reader.get_tensor("vgg_16/conv1/conv1_1/biases")
			init_biases["conv1_2"] = reader.get_tensor("vgg_16/conv1/conv1_2/biases")
			init_biases["conv2_1"] = reader.get_tensor("vgg_16/conv2/conv2_1/biases")
			init_biases["conv2_2"] = reader.get_tensor("vgg_16/conv2/conv2_2/biases")
			init_biases["conv3_1"] = reader.get_tensor("vgg_16/conv3/conv3_1/biases")
			init_biases["conv3_2"] = reader.get_tensor("vgg_16/conv3/conv3_2/biases")
			init_biases["conv3_3"] = reader.get_tensor("vgg_16/conv3/conv3_3/biases")
			init_biases["conv4_1"] = reader.get_tensor("vgg_16/conv4/conv4_1/biases")
			init_biases["conv4_2"] = reader.get_tensor("vgg_16/conv4/conv4_2/biases")
			init_biases["conv4_3"] = reader.get_tensor("vgg_16/conv4/conv4_3/biases")
			init_biases["conv5_1"] = reader.get_tensor("vgg_16/conv5/conv5_1/biases")
			init_biases["conv5_2"] = reader.get_tensor("vgg_16/conv5/conv5_2/biases")
			init_biases["conv5_3"] = reader.get_tensor("vgg_16/conv5/conv5_3/biases")

			# load weights of all layers
			init_weights["conv1_1"] = reader.get_tensor("vgg_16/conv1/conv1_1/weights")
			init_weights["conv1_2"] = reader.get_tensor("vgg_16/conv1/conv1_2/weights")
			init_weights["conv2_1"] = reader.get_tensor("vgg_16/conv2/conv2_1/weights")
			init_weights["conv2_2"] = reader.get_tensor("vgg_16/conv2/conv2_2/weights")
			init_weights["conv3_1"] = reader.get_tensor("vgg_16/conv3/conv3_1/weights")
			init_weights["conv3_2"] = reader.get_tensor("vgg_16/conv3/conv3_2/weights")
			init_weights["conv3_3"] = reader.get_tensor("vgg_16/conv3/conv3_3/weights")
			init_weights["conv4_1"] = reader.get_tensor("vgg_16/conv4/conv4_1/weights")
			init_weights["conv4_2"] = reader.get_tensor("vgg_16/conv4/conv4_2/weights")
			init_weights["conv4_3"] = reader.get_tensor("vgg_16/conv4/conv4_3/weights")
			init_weights["conv5_1"] = reader.get_tensor("vgg_16/conv5/conv5_1/weights")
			init_weights["conv5_2"] = reader.get_tensor("vgg_16/conv5/conv5_2/weights")
			init_weights["conv5_3"] = reader.get_tensor("vgg_16/conv5/conv5_3/weights")

			# load weights and biases of fully connected layers
			fc6_biases = reader.get_tensor("vgg_16/fc6/biases")
			fc6_weights = reader.get_tensor("vgg_16/fc6/weights")
			fc7_biases = reader.get_tensor("vgg_16/fc7/biases")
			fc7_weights = reader.get_tensor("vgg_16/fc7/weights")

			# decimate weights for first fc layer
			biases = np.zeros((1024,))
			weights = np.zeros((3, 3, 512, 1024))
			for ii in range(1024):
				biases[ii] = fc6_biases[4 * ii]
				for yy in range(3):
					for xx in range(3):
						weights[yy, xx, :, ii] = fc6_weights[3 * yy, 3 * xx, :, 4 * ii]

			init_biases["conv6"] = biases
			init_weights["conv6"] = weights

			# decimate weights for second fc layer
			biases = np.zeros((1024,))
			weights = np.zeros((1, 1, 1024, 1024))
			for ii in range(1024):
				biases[ii] = fc7_biases[4 * ii]
				for jj in range(1024):
					weights[:, :, jj, ii] = fc7_weights[:, :, 4 * jj, 4 * ii]

			init_biases["conv7"] = biases
			init_weights["conv7"] = weights

			# define network name
			network_name = self.network_name
			if network_name is None:
				network_name = "ssd_vgg"

			ops = []

			# create operations for restoring biases
			for name, bias in init_biases.items():
				variable_name = "{}/vgg_16/{}/bias:0".format(network_name, name)
				for variable in tf.global_variables():
					if variable.name == variable_name:
						break
				ops.append(variable.assign(bias))

			# create operations for restoring weights
			for name, weight in init_weights.items():
				variable_name = "{}/vgg_16/{}/weights:0".format(network_name, name)
				for variable in tf.global_variables():
					if variable.name == variable_name:
						break
				ops.append(variable.assign(weight))

		return ops


	def construct_vgg_layers(self, training = True):
		"""Build layers of VGG 16 network.

		Arguments:
			training: Flag whether to build the model in train mode. Defaults to True.
		"""
		# build vgg 16 network
		with tf.variable_scope("vgg_16"):
			net = self.image_input

			net = convolution2d(net, 64, [3, 3], [1, 1], seed=self.op_seed, name="conv1_1")
			net = convolution2d(net, 64, [3, 3], [1, 1], seed=self.op_seed, name="conv1_2")
			net = max_pooling2d(net, [2, 2], [2, 2], name="pool_1")

			net = convolution2d(net, 128, [3, 3], [1, 1], seed=self.op_seed, name="conv2_1")
			net = convolution2d(net, 128, [3, 3], [1, 1], seed=self.op_seed, name="conv2_2")
			net = max_pooling2d(net, [2, 2], [2, 2], name="pool_2")

			net = convolution2d(net, 256, [3, 3], [1, 1], seed=self.op_seed, name="conv3_1")
			net = convolution2d(net, 256, [3, 3], [1, 1], seed=self.op_seed, name="conv3_2")
			net = convolution2d(net, 256, [3, 3], [1, 1], seed=self.op_seed, name="conv3_3")
			net = max_pooling2d(net, [2, 2], [2, 2], name="pool_3")

			net = convolution2d(net, 512, [3, 3], [1, 1], seed=self.op_seed, name="conv4_1")
			net = convolution2d(net, 512, [3, 3], [1, 1], seed=self.op_seed, name="conv4_2")
			net = convolution2d(net, 512, [3, 3], [1, 1], seed=self.op_seed, name="conv4_3")
			self.feature_layers["block_4"] = net
			net = max_pooling2d(net, [2, 2], [2, 2], name="pool_4")

			net = convolution2d(net, 512, [3, 3], [1, 1], seed=self.op_seed, name="conv5_1")
			net = convolution2d(net, 512, [3, 3], [1, 1], seed=self.op_seed, name="conv5_2")
			net = convolution2d(net, 512, [3, 3], [1, 1], seed=self.op_seed, name="conv5_3")
			net = max_pooling2d(net, [3, 3], [1, 1], name="pool_5")

			net = atrous_convolution2d(net, 1024, [3, 3], 6, seed=self.op_seed, name="conv6")
			if training:
				net = tf.nn.dropout(net, self.hyperparams["keep_probability"], seed=self.op_seed, name="dropout6")

			net = convolution2d(net, 1024, [1, 1], [1, 1], seed=self.op_seed, name="conv7")
			self.feature_layers["block_7"] = net
			if training:
				net = tf.nn.dropout(net, self.hyperparams["keep_probability"], seed=self.op_seed, name="dropout7")

		self.ssd_inputs = net


	def construct_extra_feature_layers(self, training = True):
		"""Build extra feature layers of SSD network.

		Arguments:
			training: Flag whether to build the model in train mode. Defaults to True.
		"""
		is_ssd_512 = False
		if len(self.hyperparams["feature_layers"]) > 6:
			is_ssd_512 = True

		net = self.ssd_inputs

		net = convolution2d(net, 256, [1, 1], [1, 1], seed=self.op_seed, name="conv8_1")
		net = convolution2d(net, 512, [3, 3], [2, 2], seed=self.op_seed, name="conv8_2")
		self.feature_layers["block_8"] = net

		net = convolution2d(net, 128, [1, 1], [1, 1], seed=self.op_seed, name="conv9_1")
		net = convolution2d(net, 256, [3, 3], [2, 2], seed=self.op_seed, name="conv9_2")
		self.feature_layers["block_9"] = net

		if not is_ssd_512:
			net = convolution2d(net, 128, [1, 1], [1, 1], seed=self.op_seed, name="conv10_1")
			net = convolution2d(net, 256, [3, 3], [1, 1], seed=self.op_seed, padding="VALID", name="conv10_2")
		else:
			net = convolution2d(net, 128, [1, 1], [1, 1], seed=self.op_seed, name="conv10_1")
			net = convolution2d(net, 256, [3, 3], [2, 2], seed=self.op_seed, name="conv10_2")
		self.feature_layers["block_10"] = net

		net = convolution2d(net, 128, [1, 1], [1, 1], seed=self.op_seed, name="conv11_1")
		net = convolution2d(net, 256, [3, 3], [1, 1], seed=self.op_seed, padding="VALID", name="conv11_2")
		self.feature_layers["block_11"] = net

		if is_ssd_512:
			net = convolution2d(net, 128, [1, 1], [1, 1], seed=self.op_seed, name="conv12_1")
			net = tf.pad(net, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="CONSTANT")
			net = convolution2d(net, 256, [3, 3], [1, 1], seed=self.op_seed, padding="VALID", name="conv12_2")
			self.feature_layers["block_12"] = net


	def construct_multibox_layers(self, training = True):
		"""Build MultiBox layers of SSD network.

		Arguments:
			training: Flag whether to build the model in train mode. Defaults to True.
		"""
		num_classes = self.hyperparams["num_classes"]
		feature_layers = self.hyperparams["feature_layers"]
		feature_shapes = self.hyperparams["feature_shapes"]
		feature_normalizations = self.hyperparams["feature_normalizations"]
		anchor_ratios = self.hyperparams["anchor_ratios"]

		with tf.variable_scope("multibox"):
			logits = []
			predictions = []
			localizations = []

			# iterate through feature layers
			for ii, feature_layer in enumerate(feature_layers):
				with tf.variable_scope(feature_layers[ii]):
					feature_shape = feature_shapes[ii]
					feature_normalization = feature_normalizations[ii]
					num_anchors = len(anchor_ratios[ii]) + 2

					# perform l2 normalization
					inputs = self.feature_layers[feature_layer]
					if feature_normalization is not None:
						inputs = l2_normalization(inputs, initial_scale=feature_normalization, name="l2_norm")

					batch_size = tf.shape(inputs)[0] # ()

					# create convolutional layer for class predictions
					num_class_predictions = num_classes * num_anchors # ()
					conv_class = convolution2d(
						inputs,
						num_class_predictions,
						[3, 3],
						[1, 1],
						activation = False,
						seed = self.op_seed,
						name = "class"
					)
					conv_class = tf.reshape(conv_class, (batch_size, -1, num_classes)) # (N, num_aboxes_layer, num_classes)

					# create convolutional layer for localization predictions
					num_location_predictions = 4 * num_anchors # ()
					conv_location = convolution2d(
						inputs,
						num_location_predictions,
						[3, 3],
						[1, 1],
						activation = False,
						seed = self.op_seed,
						name = "location"
					)
					conv_location = tf.reshape(conv_location, (batch_size, -1, 4)) # (N, num_aboxes_layer, 4)

				logits.append(conv_class)
				predictions.append(tf.nn.softmax(conv_class))
				localizations.append(conv_location)

			# flatten output tensors
			self.logits = tf.concat(logits, axis=1, name="logits") # (N, num_aboxes, num_classes)
			self.predictions = tf.concat(predictions, axis=1, name="predictions") # (N, num_aboxes, num_classes)
			self.localizations = tf.concat(localizations, axis=1, name="localizations") # (N, num_aboxes, 4)


	def construct_loss(self, weight_decay = 0.0005, name = None):
		"""Build loss layer of SSD network.

		Arguments:
			weight_decay: Weight of the L2 losses. Defaults to 1e-4.
			name: Name for variable scope of the operation. Defaults to None.
		"""
		with tf.variable_scope(name, default_name="loss"):
			batch_size = tf.shape(self.logits)[0]

			num_anchors = self.hyperparams["num_anchors"]
			num_classes = self.hyperparams["num_classes"]

			# match counters
			num_total = tf.ones([batch_size], dtype=tf.int64) * tf.cast(num_anchors, dtype=tf.int64) # (N,)
			num_positives = tf.count_nonzero(self.groundtruth_classes, axis=1) # (N,)
			num_negatives = num_total - num_positives # (N,)

			positives_num_safe = tf.where(
				tf.equal(num_positives, 0),
				tf.ones([batch_size]) * 10e-15,
				tf.to_float(num_positives)
			) # (N,)

			# match masks
			mask_negatives = tf.equal(self.groundtruth_classes, 0) # (N, num_aboxes)
			mask_positives = tf.logical_not(mask_negatives) # (N, num_aboxes)

			# confidence loss
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits = self.logits,
				labels = self.groundtruth_classes
			) # (N, num_aboxes)

			positives = tf.where(mask_positives, cross_entropy, tf.zeros_like(cross_entropy)) # (N, num_aboxes)
			sum_positives = tf.reduce_sum(positives, axis=-1) # (N,)

			negatives = tf.where(mask_negatives, cross_entropy, tf.zeros_like(cross_entropy)) # (N, num_aboxes)
			top_negatives = tf.nn.top_k(negatives, num_anchors)[0] # (N, num_aboxes)

			num_max_negatives = tf.minimum(num_negatives, 3 * num_positives) # (N,)
			num_max_negatives_t = tf.expand_dims(num_max_negatives, 1) # (N, 1)

			rng = tf.range(num_anchors) # (num_aboxes,)
			rng_row = tf.cast(tf.expand_dims(rng, 0), dtype=tf.int64) # (1, num_aboxes)

			mask_max_negatives = tf.less(rng_row, num_max_negatives_t) # (N, num_aboxes)
			max_negatives = tf.where(mask_max_negatives, top_negatives, tf.zeros_like(top_negatives)) # (N, num_aboxes)

			sum_max_negatives = tf.reduce_sum(max_negatives, axis=-1) # (N,)

			confidence_loss = tf.add(sum_positives, sum_max_negatives) # (N,)
			confidence_loss = tf.where(
				tf.equal(num_positives, 0),
				tf.zeros([batch_size]),
				tf.div(confidence_loss, positives_num_safe)
			) # (N,)

			self.losses["confidence"] = tf.reduce_mean(confidence_loss, name="confidence_loss") # ()

			# localization loss
			localization_difference = tf.subtract(self.localizations, self.groundtruth_localizations) # (N, num_aboxes, 4)

			localization_loss = smooth_l1_loss(localization_difference) # (N, num_aboxes, 4)

			sum_localization_loss = tf.reduce_sum(localization_loss, axis=-1) # (N, num_aboxes)

			positive_localizations = tf.where(
				mask_positives,
				sum_localization_loss,
				tf.zeros_like(sum_localization_loss)
			) # (N, num_aboxes)

			localization_loss = tf.reduce_sum(positive_localizations, axis=-1) # (N,)
			localization_loss = tf.where(
				tf.equal(num_positives, 0),
				tf.zeros([batch_size]),
				tf.div(localization_loss, positives_num_safe)
			) # (N,)

			self.losses["localization"] = tf.reduce_mean(localization_loss, name="localization_loss") # ()

			# regularization loss
			l2_losses = tf.get_collection("l2_losses")
			l2_loss = tf.add_n(l2_losses)
			self.losses["l2"] = tf.multiply(weight_decay, l2_loss, name="l2_loss")

			# total loss
			self.losses["total"] = tf.add_n([
				self.losses["confidence"],
				self.losses["localization"],
				self.losses["l2"]
			], name="total_loss")


	def construct_post_processing(self):
		"""Build post-processing layer for the SSD network.
		"""
		with tf.variable_scope("output"):
			# decode SSD predicted boxes
			decoded_boxes = tf_ssd_decode_boxes(
				self.localizations,
				self.get_default_anchor_boxes(),
				self.hyperparams["prior_scaling"]
			)

			# select predicted boxes
			c_classes, c_scores, c_localizations = tf_ssd_select_boxes(
				self.predictions,
				decoded_boxes,
				confidence_threshold = 0.5,
				top_k = 200
			)

			# filter predicted boxes using nms
			c_classes, c_scores, c_localizations = tf_ssd_suppress_overlaps(
				c_classes,
				c_scores,
				c_localizations,
				keep_top_k = 200,
				nms_threshold = 0.45
			)

			self.output = {
				"classes": tf.identity(c_classes, name="classes"),
				"scores": tf.identity(c_scores, name="scores"),
				"localizations": tf.identity(c_localizations, name="localizations")
			}
