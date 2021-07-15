#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:11:13 2021

@author: shesriva
"""

# Face Recognition using KNN
# run using command: python construct_data_matrices.py training_data.npy training_labels.npy label_dict.json test_data.npy test_labels.npy
# ------------------------------------------------------------------------------ #

# Generating Combined Data

import numpy as np
import os
import sys
import json

# current_path = '/Users/shesriva/Desktop/DS/faceRecUsingKNN'
data_path = '../raw_data' # Update the relative path of the data here.

# Feature Matrix Initialisation
X = np.array([])
flattened_image_size = 67500
X.shape = (0, flattened_image_size)

# Labels Initialisation
y = np.array([])
y.shape = (0, 1)

X_test = np.array([])
X_test.shape = (0, flattened_image_size)

y_test = np.array([])
y_test.shape = (0, 1)

# Assign a class to every .npy file encountered
current_label = 0

# Label to Person dictionary
label_dict = {}

for filename in os.listdir(data_path):
    if filename.endswith(".npy"):
        
        # Load Data
        all_data = np.load(data_path + "/" + filename)
        
        # Get Name of Person from filename format: PERSONNAME.npy
        person_name = filename.split(".npy")[0]
        
        # print(data.shape)
        
        data = all_data[:30] # Restrict data to first 30 instances only.
        test_data = all_data[30:]
    
        # Add to Feature Matrix
        X = np.concatenate([X, data], axis = 0)
        X_test = np.concatenate([X_test, test_data], axis = 0)
        
        # Compute current labels
        current_y = np.ones(data.shape[0]).reshape(-1, 1) * current_label
        y = np.concatenate([y, current_y], axis = 0)
        
        current_y_test = np.ones(test_data.shape[0]).reshape(-1, 1) * current_label
        y_test = np.concatenate([y_test, current_y_test], axis = 0)
        
        # Save label vs person_name in dict for mapping predicted labels to person names.
        label_dict[current_label] = person_name
        
        # Increment label
        current_label += 1

# Typecase labels into integers : These are float by default.
y = y.astype('int')
print('Shape of Training Data = ', X.shape)
print('Shape of Labels of Training Data = ', y.shape)

# Feature Matrix with size (n_sampes, flattened_image_size) is ready.
# Labels Matrix with size (n_samples, 1) is ready.
# Label to Person mapping is stored in dictionary.

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)

filename_X = sys.argv[1]
filename_y = sys.argv[2]
np.save("../training_data/" + filename_X, X)
np.save("../training_data/" + filename_y, y)

dict_name = sys.argv[3]

with open("../training_data/" + dict_name, 'w') as fp:
    json.dump(label_dict, fp)

y_test = y_test.astype('int')
print('Shape of Test Data = ', X_test.shape)
print('Shape of Labels of Test Data = ', y_test.shape)

from sklearn.utils import shuffle
X_test, y_test = shuffle(X_test, y_test, random_state=0)

filename_X_test = sys.argv[4]
filename_y_test = sys.argv[5]
np.save("../test_data/" + filename_X_test, X_test)
np.save("../test_data/" + filename_y_test, y_test)


# ------------------------------------------------------------------------------ #


