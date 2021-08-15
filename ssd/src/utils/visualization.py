##
## /src/utils/visualization.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 15/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import colorsys
import cv2
import numpy as np
import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


def get_distinct_colors(n):
	"""Compute a number of distinct colors.

	Arguments:
		n: Number of colors to compute.

	Returns:
		List of computed colors.
	"""
	partition = 1.0 / (n + 1)
	colors = [colorsys.hsv_to_rgb(partition * ii, 1.0, 0.6) for ii in range(n)]
	for ii in range(n):
		colors[ii] = (
			int(255.0 * colors[ii][0]),
			int(255.0 * colors[ii][1]),
			int(255.0 * colors[ii][2])
		)
	return colors


def draw_single_box(image, # (h, w, c)
					box, # (4,) <- (ymin, xmin, ymax, xmax)
					score = None, # ()
					label = None, # ()
					color = (0, 0, 0)): # (r, g, b)
	"""Draw a single bounding box on an image.

	 Arguments:
	 	image: Image to process.
		box: Coordinates of the bounding box to draw.
		score: Score of the bounding box. Defaults to None.
		label: Label of the bounding box. Defaults to None.
		color: RGB color of the bounding box. Defaults to (0, 0, 0).
	"""
	# convert rgb color to bgr color
	bgr_color = (color[2], color[1], color[0])

	image_height = image.shape[0]
	image_width = image.shape[1]

	# compute absolute coordinates of box
	y_min = np.maximum(box[0] * image_height, 0.0).astype(np.uint32)
	x_min = np.maximum(box[1] * image_width, 0.0).astype(np.uint32)
	y_max = np.minimum(box[2] * image_height, image_height).astype(np.uint32)
	x_max = np.minimum(box[3] * image_width, image_width).astype(np.uint32)

	# draw bounding box
	image_box = np.copy(image)
	cv2.rectangle(image_box, (x_min, y_min), (x_max, y_max), bgr_color, thickness=2)

	# draw label and / or score
	if label is not None or score is not None:
		font = cv2.FONT_HERSHEY_DUPLEX
		if label is None:
			text = "{:.2f}".format(score)
		elif score is None:
			text = "{}".format(label)
		else:
			text = "{}: {:.2f}".format(label, score)

		# compute location of text
		text_size = cv2.getTextSize(text, font, 0.5, 1)
		text_box_height = text_size[0][1] + 10
		text_box_width = text_size[0][0] + 10
		if y_min < text_box_height:
			y_min = text_box_height
		if x_min > image_width - text_box_width:
			x_min = image_width - text_box_width

		# draw background box
		cv2.rectangle(
			image_box,
			(x_min - 1, y_min - text_box_height),
			(x_min + text_box_width, y_min),
			bgr_color,
			thickness = cv2.FILLED
		)

		# draw text
		cv2.putText(image_box, text, (x_min + 5, y_min - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

	# draw result with opacity
	alpha = 0.8
	cv2.addWeighted(image_box, alpha, image, 1.0 - alpha, 0, image)


def draw_fps(image, # (h, w, c)
			 fps): # ()
	"""Draw a FPS value on an image.

	Arguments:
		image: Image to process.
		fps: FPS value to draw.
	"""
	font = cv2.FONT_HERSHEY_DUPLEX
	text = "{:.2f}".format(fps)

	image_fps = np.copy(image)
	text_size = cv2.getTextSize(text, font, 0.5, 1)

	# compute location of text
	y_min = 10
	x_min = 10
	y_max = y_min + text_size[0][1] + 10
	x_max = x_min + text_size[0][0] + 10

	# draw background and text
	cv2.rectangle(image_fps, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=cv2.FILLED)
	cv2.putText(image_fps, text, (x_min + 5, y_max - 6), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

	# draw result with opacity
	alpha = 0.8
	cv2.addWeighted(image_fps, alpha, image, 1.0 - alpha, 0, image)
