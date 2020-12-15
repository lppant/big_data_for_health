import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools as iter


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	figure, subplots = plt.subplots(1, 2, figsize=(20, 10))

	subplots[0].set_title('Loss Curve')
	subplots[0].set_xlabel("epoch")
	subplots[0].set_ylabel("Loss")
	subplots[0].plot(train_losses, 'C0', label='Training Loss')
	subplots[0].plot(valid_losses, 'C1', label='Validation Loss')
	subplots[0].legend(loc="upper right")

	subplots[1].set_title('Accuracy Curve')
	subplots[1].set_xlabel("epoch")
	subplots[1].set_ylabel("Accuracy")
	subplots[1].plot(train_accuracies, 'C0', label='Training Accuracy')
	subplots[1].plot(valid_accuracies, 'C1', label='Validation Accuracy')
	subplots[1].legend(loc="upper left")

	figure.savefig('Learning_Curves.png')

#Citiations: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(results, class_names):
	y_true, y_pred = zip(* results)
	conf_mat = confusion_matrix(y_true, y_pred)
	np.set_printoptions(precision = 2)
	plt.figure()
	conf_mat = conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]
	plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Normalized Confusion Matrix")
	plt.ylabel('True')
	plt.xlabel('Predicted')
	plt.colorbar()
	ticks = np.arange(len(class_names))
	plt.xticks(ticks, class_names, rotation=45)
	plt.yticks(ticks, class_names)

	# Iterate over data dimensions to place text output
	fmt = '.2f'
	threshold = conf_mat.max() / 2.
	for x, y in iter.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
		plt.text(y, x, format(conf_mat[x, y], fmt),
				 horizontalalignment="center",
				 color="white" if conf_mat[x, y] > threshold else "black")
	plt.tight_layout()
	plt.savefig("Confusion_Matrix.png")