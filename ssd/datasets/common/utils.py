##
## /src/datasets/common/utils.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 15/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from datasets import VOC2007, VOC2007_2012, VOC2012


def get_dataset(dataset_name, *args, **kwargs):
	"""Gets a dataset object by name.

	Arguments:
		dataset_name: Name of the dataset.
		args: List of arguments passed to constructor of dataset class.
		kwargs: Dictionary of arguments passed to constructor of dataset class.
	"""
	dataset_name = dataset_name.lower()
	if dataset_name == "voc2007":
		dataset = VOC2007
	elif dataset_name == "voc2007+2012":
		dataset = VOC2007_2012
	elif dataset_name == "voc2012":
		dataset = VOC2012
	else:
		raise RuntimeError("The dataset '{}' is currently not supported.".format(dataset_name))

	return dataset(*args, **kwargs)
