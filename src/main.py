#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:55:18 2021

@author: shesriva
"""

from knn_classifier import KNNClassifier
import numpy as np
import sys
import json

# Load Training Data
data_path = "../training_data/" + sys.argv[1]
labels_path = "../training_data/" + sys.argv[2]
X = np.load(data_path)
y = np.load(labels_path)

# Load Label Dict
label_dict_path = "../training_data/" + sys.argv[3]
with open("../training_data/" + sys.argv[3], 'r') as fp:
    label_dict = json.load(fp)

# Load Test Data
X_test_path = "../test_data/" + sys.argv[4]
X_test = np.load(X_test_path)
y_test_path = "../test_data/" + sys.argv[5]
y_test = np.load(y_test_path)

# Create Classifier
knn = KNNClassifier()
knn.fit(X, y)

# Predict on X_test
predictions = knn.predict(X_test) # Save Prediction Name Also!

predictions = np.array(predictions).reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

for i in range(len(predictions)):
    predicted_label_value = predictions[i][0]
    predicted_label = label_dict[str(predictions[i][0])]
    true_label = y_test[i][0]
    print('Predicted Label = ', predicted_label, 'True Label = ', label_dict[str(true_label)])
    # print('True Label = ', true_label)
    # if (predicted_label_value == true_label):
    #     print('This one was correctly predicted!')
    # else:
    #     print('Incorrect!')

acc = knn.accuracy(predictions, y_test)
print('Accuracy of Network = ', acc)
