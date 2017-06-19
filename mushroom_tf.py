from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import csv
import random
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
M_TRAIN = 'm_train.csv'
M_TEST = 'm_test.csv'

def make_train_and_test_CSV():
	"""
	Randomly sets aside about 10% of the data to be in a test set,
	while leaving the rest as a training set.
	"""
	ignore = True
	first = True
	testIndices = random.sample(range(1, 8126), 813)
	counter = 1
	with open('mushroom.csv', newline='') as csvfile:
		mreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in mreader:
			if ignore:
				ignore = False
				continue
			if first:
				trainList, testList = [row], [row]
				first = False
				continue
			if counter in testIndices:
				testList.append(row)
			else:
				trainList.append(row)
			counter += 1
		with open(M_TRAIN, 'w', newline='') as fp:
			a = csv.writer(fp, delimiter='|')
			a.writerows(trainList)
		with open(M_TEST, 'w', newline='') as fp:
			a = csv.writer(fp, delimiter='|')
			a.writerows(testList)

def load_sets_for_tf():
	"""
	Loads the appropriate files ready for tf usage.
	"""
	training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
	    filename=M_TRAIN,
	    target_dtype=np.int,
	    features_dtype=np.int,
	    target_column=0)

	test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
	    filename=M_TEST,
	    target_dtype=np.int,
	    features_dtype=np.int,
	    target_column=0)
	return training_set, test_set

def get_train_inputs():
	"""
	Gets the training set's data and target.
	"""
	x = tf.constant(training_set.data)
	y = tf.constant(training_set.target)
	return x, y

def get_test_inputs():
	"""
	Gets the test set's data and target.
	"""
	x = tf.constant(test_set.data)
	y = tf.constant(test_set.target)
	return x, y

if __name__ == "__main__":
    make_train_and_test_CSV()
    training_set, test_set = load_sets_for_tf()
    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=22)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
			                                    hidden_units=[10, 20, 10],
			                                    n_classes=2)
    classifier.fit(input_fn=get_train_inputs, steps=2000)
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)['accuracy']
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



















