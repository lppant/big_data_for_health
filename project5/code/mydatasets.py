import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""

	seizure_df = pd.read_csv(path)

	if model_type == 'MLP':
		data = torch.tensor(seizure_df.drop('y', axis = 1).values.astype('float32'))
		target = torch.tensor((seizure_df['y'] - 1).values.astype('long'))
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = torch.tensor(seizure_df.drop('y', axis = 1).values.astype('float32'))
		target = torch.tensor((seizure_df['y'] - 1).values.astype('long'))
		dataset = TensorDataset(data.unsqueeze(1), target)
	elif model_type == 'RNN':
		data = torch.tensor(seizure_df.drop('y', axis = 1).values.astype('float32'))
		target = torch.tensor((seizure_df['y'] - 1).values.astype('long'))
		dataset = TensorDataset(data.unsqueeze(2), target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# As seqs is a list of list of list, we can apply 2 reduce function to get all features.
	visits = reduce(lambda x1, x2: x1 + x2, seqs)
	diag_codes = reduce(lambda x1, x2: x1 + x2, visits)

	# Then, get distinct values from features using set and find the length of the resulting set.
	features = len(set(diag_codes))
	return features


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels
		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		#self.seqs = [i for i in range(len(labels))]  # replace this with your implementation.
		sequences = []
		for seq in seqs:
			row = 0
			arr = np.zeros((len(seq), num_features))
			for elements in seq:
				for element in elements:
					arr[row, element] = 1
				row = row + 1
			sequences.append(arr)
		self.seqs = sequences

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence

	# seqs_tensor = torch.FloatTensor()
	# lengths_tensor = torch.LongTensor()
	# labels_tensor = torch.LongTensor()
	#
	# return (seqs_tensor, lengths_tensor), labels_tensor

	def collateSequences(batch, tuple, feature):
		seqs, lengths, labels = ([] for i in range(3))
		for i in map(lambda s: s[1], tuples):
			arr = np.zeros((tuple, feature))
			entity = batch[i]
			lengths.append(entity[0].shape[0])
			labels.append(entity[1])
			arr[0:entity[0].shape[0], 0:entity[0].shape[1]] = entity[0]
			seqs.append(arr)
		return (seqs, lengths, labels)

	tuples = []
	pos = 0
	for seq,label in batch:
		tuples.append((seq.shape[0], pos))
		pos += 1
	tuples.sort(key=lambda v: v[0], reverse=True)
	tuple = tuples[0][0]
	feature = batch[0][0].shape[1]
	collated = collateSequences(batch, tuple, feature)

	seqs_tensor = torch.FloatTensor(collated[0])
	lengths_tensor = torch.LongTensor(collated[1])
	labels_tensor = torch.LongTensor(collated[2])

	return (seqs_tensor, lengths_tensor), labels_tensor
