##
## /src/models/ssd/common/utils.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 15/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import os
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from models.ssd import SSDVGG300
from models.ssd import SSDVGG512


def get_model(model_name, *args, **kwargs):
	"""Gets a model object by name.

	Arguments:
		model_name: Name of the model.
		args: List of arguments passed to constructor of model class.
		kwargs: Dictionary of arguments passed to constructor of model class.
	"""
	model_name = model_name.lower()
	if model_name == "ssd_vgg_300":
		model = SSDVGG300
	elif model_name == "ssd_vgg_512":
		model = SSDVGG512
	else:
		raise RuntimeError("The model '{}' is currently not supported.".format(model_name))

	return model(*args, **kwargs)
