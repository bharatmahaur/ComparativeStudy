##
## /src/utils/average_precision.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 08/07/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 26/07/2018.
##

import math
import numpy as np
import os
import sys

sys.path.append(sys.path[0])
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models.ssd.common import np_jaccard_score
from utils.common.files import get_full_path


class AveragePrecision(object):
	"""Handle the calculation of average precision values by class.

	Arguments:
		matching_threshold: Jaccard score threshold used for matches.
		counts_by_class: Dictionary containing number of groundtruth boxes per class.
		gboxes_by_class: Dictionary containing groundtruth boxes per class.
		predictions_by_class: Dictionary containing predicted boxes and scores per class.
		last_sample_id: Last remembered sample index.
		__clean: Flag whether the state of the class is clean.
	"""

	def __init__(self, matching_threshold = 0.5):
		"""Initialize the class.

		Arguments:
			matching_threshold: Jaccard score threshold used for matches.
		"""
		self.matching_threshold = matching_threshold

		self.counts_by_class = {} # { class: num_gboxes }
		self.gboxes_by_class = {} # { class: { sample_id: tuple(localizations, matches) } }
		self.predictions_by_class = {} # { class: [tuple(localization, score, sample_id)] }
		self.last_sample_id = 0
		self.__clean = True


	def add_predictions(self,
						groundtruth_classes, # (N, num_gboxes)
						groundtruth_boxes, # (N, num_gboxes, 4) <- (ymin, xmin, ymax, xmax)
						predicted_classes, # (N, num_predictions)
						predicted_scores, # (N, num_predictions)
						predicted_localizations): # (N, num_predictions, 4) <- (ymin, xmin, ymax, xmax)
		"""Add new predictions for average precision calculation.

		Arguments:
			groundtruth_classes: Tensor of shape (N, num_gboxes) containing groundtruth classes.
			groundtruth_boxes: Tensor of shape (N, numb_gboxes, 4) containing groundtruth boxes.
			predicted_classes: Tensor of shape (N, num_predictions) containing predicted classes.
			predicted_scores: Tensor of shape (N, num_predictions) containing predicted scores.
			predicted_localizations: Tensor of shape (N, num_predictions, 4) containing predicted localizations.
		"""
		num_samples = groundtruth_classes.shape[0]
		num_gboxes = groundtruth_classes.shape[1]
		num_predictions = predicted_classes.shape[1]

		# iterate through samples
		for ii in range(num_samples):
			boxes_by_class = {} # { class: [localization] }

			# iterate through groundtruth boxes and sort scores by class
			for jj in range(num_gboxes):
				gclass = groundtruth_classes[ii, jj]
				if gclass == 0:
					continue
				self.counts_by_class.setdefault(gclass, 0)
				self.counts_by_class[gclass] += 1
				boxes_by_class.setdefault(gclass, []).append(groundtruth_boxes[ii, jj])

			# sort groundtruth boxes by class
			for cl, localizations in boxes_by_class.items():
				self.gboxes_by_class.setdefault(cl, {})[self.last_sample_id] = (
					np.stack(localizations, axis=0), np.zeros((len(localizations),), dtype=np.bool)
				)

			# iterate through predicted boxes and sort by class
			for jj in range(num_predictions):
				pclass = predicted_classes[ii, jj]
				if pclass == 0:
					continue
				self.predictions_by_class.setdefault(pclass, []).append(
					(predicted_localizations[ii, jj], predicted_scores[ii, jj], self.last_sample_id)
				)

			# increase sample index
			self.last_sample_id += 1

		self.__clean = False


	def precision_recall(self):
		"""Compute the precision and recall by class.

		Returns:
			precision_by_class: Dictionary containing precision values by class.
			recall_by_class: Dictionary containing recall values by class.
		"""
		precision_by_class = {}
		recall_by_class = {}

		# iterate through all classes
		for cl in self.gboxes_by_class:
			if cl not in self.predictions_by_class:
				continue # the class was not predicted

			# stack prediction boxes and scores to an array
			plocalizations = []
			pscores = []
			psample_ids = []
			for prediction in self.predictions_by_class[cl]:
				plocalizations.append(prediction[0])
				pscores.append(prediction[1])
				psample_ids.append(prediction[2])
			plocalizations = np.stack(plocalizations, axis=0)
			pscores = np.stack(pscores, axis=0)
			psample_ids = np.stack(psample_ids, axis=0)

			# sort scores in descending order
			best_indexes = np.argsort(-pscores)
			plocalizations = plocalizations[best_indexes]
			pscores = pscores[best_indexes]
			psample_ids = psample_ids[best_indexes]

			num_predictions = best_indexes.shape[0]
			tps = np.zeros((num_predictions,), dtype=np.bool)
			fps = np.zeros((num_predictions,), dtype=np.bool)

			# iterate through predictions
			for ii in range(num_predictions):
				psample_id = psample_ids[ii]
				if psample_id not in self.gboxes_by_class[cl]:
					fps[ii] = True
					continue # the image this prediction comes from contains no objects of this class

				plocalization = plocalizations[ii]
				glocalizations = self.gboxes_by_class[cl][psample_id][0]
				gmatches = self.gboxes_by_class[cl][psample_id][1]

				jaccard = np_jaccard_score(plocalization[None, :], glocalizations)[0]
				best_index = np.argmax(jaccard)
				if jaccard[best_index] < self.matching_threshold:
					fps[ii] = True
					continue # jaccard overlap is under threshold

				if gmatches[best_index]:
					fps[ii] = True
					continue # groundtruth box with max overlap is already matched

				tps[ii] = True
				gmatches[best_index] = True

			# sum true and false positives cumulative
			fps = np.cumsum(fps)
			tps = np.cumsum(tps)
			precision = tps / (tps + fps)
			recall = tps / self.counts_by_class[cl]

			precision_by_class[cl] = precision
			recall_by_class[cl] = recall

		return precision_by_class, recall_by_class


	def average_precision(self,
						  precision, # () _or_ {cl: ()}
						  recall): # () _or_ {cl: ()}
		"""Compute average precision values by class.

		Arguments:
			precision: Dictionary containing precision values by class.
			recall: Dictionary containing recall values by class.

		Returns:
			Dictionary containing average precision values by class.
		"""
		def __average_precision(precision, recall):
			"""Compute average precision value.

			Arguments:
				precision: Single precision value.
				recall: Single recall value.

			Returns:
				Average precision value.
			"""
			precision = precision.astype(np.float64)
			recall = recall.astype(np.float64)

			precision = np.concatenate([precision, [0.0]], axis=0)
			recall = np.concatenate([recall, [np.inf]], axis=0)

			aps = []
			for tilde in np.arange(0.0, 1.1, 0.1):
				pr = precision[recall >= tilde]
				aps.append(
					np.max(pr) / 11.0
				)

			return np.sum(aps)

		# iterate through classes and compute average precision value
		if isinstance(precision, dict) or isinstance(recall, dict):
			cl_ap = {}
			for cl in precision:
				ap = __average_precision(
					precision[cl],
					recall[cl]
				)
				cl_ap[cl] = ap
			return cl_ap

		ap = __average_precision(precision, recall)
		return ap


	@property
	def is_clean(self):
		"""Flag whether the state of the class is clean.
		"""
		return self.__clean


	def clear(self):
		"""Clear the state of the class.
		"""
		self.counts_by_class = {}
		self.gboxes_by_class = {}
		self.predictions_by_class = {}
		self.last_sample_id = 0
		self.__clean = True
