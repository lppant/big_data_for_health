import os
import pickle
import pandas as pd
import numpy as np

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object).strip()
	# TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# TODO: Read the homework description carefully.
	char_to_extract = 3
	if not icd9_str[0] == 'V' and icd9_str[0].isalpha():
		char_to_extract = 4
	if char_to_extract < len(icd9_str):
		return icd9_str[0: char_to_extract]
	return icd9_str


def build_codemap(df_icd9, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.

	#codemap = {123: 0, 456: 1}

	df_icd9_codes = df_icd9['ICD9_CODE'].dropna().apply(transform)
	unique_icd9_codes = df_icd9_codes.unique()
	tupleList = [(item, index) for (index, item) in enumerate(unique_icd9_codes)]
	codemap = dict(tupleList)
	return codemap


def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# TODO: Loading the mortality file is shown as an example below. Load two other files also.
	##df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))

	##patient_ids = [0, 1, 2]
	##labels = [1, 0, 1]
	##seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]
	##return patient_ids, labels, seq_data

	# TODO: 1. Load data from the three csv files
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_diagnosis = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
	df_admissions = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))

	# TODO: 2. Convert diagnosis code in to unique feature ID.
	# TODO: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
	df_diagnosis['ICD9_CODE'] = df_diagnosis['ICD9_CODE'].transform(transform)

	# TODO: 3. Group the diagnosis codes for the same visit.
	groupedDiagnosisByVisit = df_diagnosis.groupby(['HADM_ID'])

	# TODO: 4. Group the visits for the same patient.
	groupedAdmissionByPatient = df_admissions.groupby(['SUBJECT_ID'])

	patient_ids, labels, seq_data = ([] for i in range(3))
	for patient_id, patient_visits in groupedAdmissionByPatient:
		#Create patient ids
		patient_ids.append(patient_id)
		#Create Labels
		selected_mortality = df_mortality[df_mortality['SUBJECT_ID'] == patient_id]
		label = selected_mortality['MORTALITY'].values[0]
		labels.append(label)
		#Create Sequence Data
		seq = []
		# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
		# TODO: Visits for each patient must be sorted in chronological order.
		patient_visits = patient_visits.sort_values(by=['ADMITTIME'])

		# TODO: 6. Make patient-id List and label List also.
		# TODO: The order of patients in the three List output must be consistent.
		for i, visit in patient_visits.iterrows():
			vist_diags = groupedDiagnosisByVisit.get_group(visit['HADM_ID'])['ICD9_CODE'].values
			selected_visit_diags = list(filter(lambda key: key in codemap.keys(), vist_diags))
			seq.append(list(map(lambda diag: codemap[diag], selected_visit_diags)))
		seq_data.append(seq)
	return patient_ids, labels, seq_data

def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
