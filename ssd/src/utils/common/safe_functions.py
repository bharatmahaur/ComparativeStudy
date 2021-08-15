##
## /src/utils/common/safe_functions.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 04/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


def safe_list_of_objects(input, input_class = None):
	"""Create a list given an object or a list of objects.

	Arguments:
		input: Object or list of objects.
		input_class: Class of the objects.

	Returns:
		List of objects.
	"""
	if input_class is not None:
		if not isinstance(input, (input_class, list)):
			raise Exception("The given variable should be an object of type '{}' or a list.".format(input_class))

	safe_list = [input]
	if isinstance(safe_list[0], list):
		safe_list = safe_list[0]

	return safe_list


def safe_iterator_of_objects(input, input_class = None):
	"""Create a generator given an object or a list of objects.

	Arguments:
		input: Object or list of objects.
		input_class: Class of the objects.

	Returns:
		Generator of objects.
	"""
	if input_class is not None:
		if not isinstance(input, (input_class, list)):
			raise Exception("The given variable should be an object of type '{}' or a list.".format(input_class))

	safe_list = [input]
	if isinstance(safe_list[0], list):
		safe_list = safe_list[0]

	for object in safe_list:
		yield object


def safe_list_of_callables(input, input_class = None):
	"""Create a list given a callable or a list of callables.

	Arguments:
		input: Object or list of callables.
		input_class: Class of the callables.

	Returns:
		List of callables.
	"""
	safe_list = safe_list_of_objects(input, input_class=input_class)
	for object in safe_list:
		if not callable(object):
			raise Exception("The given variable should be an object of type '{}' or a list.".format(input_class))

	return safe_list


def safe_iterator_of_callables(input, input_class = None):
	"""Create a generator given a callable or a list of callables.

	Arguments:
		input: Object or list of callables.
		input_class: Class of the callables.

	Returns:
		Generator of callables.
	"""
	safe_iterator = safe_iterator_of_objects(input, input_class=input_class)
	for object in safe_iterator:
		if not callable(object):
			raise Exception("The given variable should be an object of type '{}' or a list.".format(input_class))
		yield object
