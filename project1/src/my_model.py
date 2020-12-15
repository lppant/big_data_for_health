import utils
import etl
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import KFold, ShuffleSplit

from numpy import mean
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	deliverables_path = '../data/train/'
	test_path = '../data/test/'
	events, feature_map = read_test_data(test_path)
	aggregated_events = etl.aggregate_events(events, None, feature_map, deliverables_path)

	patient_features = {}
	for key, value in aggregated_events.groupby('patient_id'):
		patient_features[key] = list(zip(value['feature_id'], value['feature_value']))
	save_test_features(patient_features, '../deliverables/features_svmlight.test', '../deliverables/test_features.txt')

	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../deliverables/features_svmlight.test")
	return X_train, Y_train, X_test

def read_test_data(filepath):
	#Columns in events.csv - patient_id,event_id,event_description,timestamp,value
	print(sys.version)
	print(pd.__version__)
	events = pd.read_csv(filepath + 'events.csv')

	#Columns in event_feature_map.csv - idx,event_id
	feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

	return events, feature_map

def save_test_features(patient_features, op_svmlight, op_feature):
    svmlight_out = ''
    feature_out = ''
    for key in sorted(patient_features.keys()):
        feature_tuples = patient_features.get(key)
        line = ''
        sorted_by_feature_id = sorted(feature_tuples, key=lambda tuple: tuple[0])
        for feature_id, feature_value in sorted_by_feature_id:
            line += ' ' + str(int(feature_id)) + ':' + '%.6f' % feature_value
        svmlight_out += str(0) + line + ' \n'
        feature_out += str(int(key)) + line + '\n'

    deliverable1 = open(op_svmlight, 'wb')
    deliverable2 = open(op_feature, 'wb')

    deliverable1.write(bytes(svmlight_out,'UTF-8'))
    deliverable2.write(bytes(feature_out,'UTF-8'))

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	acc_array=[]
	auc_array=[]
	kf = KFold(n_splits=k)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
		acc = accuracy_score(Y_test,Y_pred)
		auc = roc_auc_score(Y_test, Y_pred)
		acc_array=np.append(acc_array, acc)
		auc_array=np.append(auc_array, auc)
	mean_acc = mean(acc_array)
	mean_auc = mean(auc_array)
	return mean_acc,mean_auc


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations

	acc_array=[]
	auc_array=[]
	kf = ShuffleSplit(n_splits=iterNo, test_size=test_percent)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
		acc = accuracy_score(Y_test,Y_pred)
		auc = roc_auc_score(Y_test, Y_pred)
		acc_array=np.append(acc_array, acc)
		auc_array=np.append(auc_array, auc)
	mean_acc = mean(acc_array)
	mean_auc = mean(auc_array)
	return mean_acc,mean_auc

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#baseModel = DecisionTreeClassifier(random_state=545510477,max_depth=10)
	baseModel = LogisticRegression()
	#model = BaggingClassifier(baseModel, n_estimators=10, max_samples=1.0, max_features=1.0, random_state=545510477)
	baseModel.fit(X_train, Y_train)
	y_pred = baseModel.predict(X_test)
	return y_pred

def main():
	# Y_test as an returned element added by me
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)

	# Calculate model performance using Accuracy and AUC from K-Fold and Randomized CV
	mean_acc_k,mean_auc_k = get_acc_auc_kfold(X_train, Y_train)
	print("______________________________________________")
	print(("Average AUC in K-Fold: "+str(mean_auc_k)))

	mean_acc_r,mean_auc_r = get_acc_auc_randomisedCV(X_train, Y_train)
	print(("Average AUC in Randomised CV: "+str(mean_acc_r)))
	print("______________________________________________")
	print("")

	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

