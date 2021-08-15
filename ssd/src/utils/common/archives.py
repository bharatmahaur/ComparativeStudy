##
## /src/utils/common/archives.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 13/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 25/07/2018.
##

import os
import shutil
import sys
import tarfile
import tempfile
import uuid
import zipfile

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.files import mkdir


def extract_archive(archive_path, destination_path):
	"""Extract an archive to a given destination directory.

	Arguments:
		archive_path: Path to the archive.
		destination_path: Destination directory where to extract the archive.
	"""
	archive_filename = os.path.basename(archive_path)

	# create temporary directory for extracting archive
	temp_filename = "{}_{}".format(archive_filename, str(uuid.uuid4()))
	temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
	mkdir(temp_path)

	# extract archive
	if archive_filename.endswith(".zip"):
		archive_name = archive_filename[:-4]
		with zipfile.ZipFile(archive_path, "r") as file:
			file.extractall(temp_path)
	elif archive_filename.endswith(".tar"):
		archive_name = archive_filename[:-4]
		with tarfile.open(archive_path, "r:") as file:
			file.extractall(temp_path)
	elif archive_filename.endswith(".tar.bz2"):
		archive_name = archive_filename[:-8]
		with tarfile.open(archive_path, "r:bz2") as file:
			file.extractall(temp_path)
	elif archive_filename.endswith(".tar.gz"):
		archive_name = archive_filename[:-7]
		with tarfile.open(archive_path, "r:gz") as file:
			file.extractall(temp_path)
	else:
		raise NotImplementedError("The type of the archive '{}' is currently not supported.".format(archive_filename))

	# build target path
	target_path = os.path.join(destination_path, archive_name)

	# move extracted files to destination
	extracted_items = os.listdir(temp_path)
	if len(extracted_items) == 1 and extracted_items[0] == archive_name:
		single_path = os.path.join(temp_path, archive_name)
		if os.path.isdir(single_path):
			shutil.move(single_path, target_path)
			return

	shutil.move(temp_path, target_path)
