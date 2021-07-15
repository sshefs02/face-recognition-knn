#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:48:58 2021

@author: shesriva
"""

# ------------------------------------------------------------------------------ #

# KNN Classifier

# ------------------------------------------------------------------------------ #

import numpy as np

class KNNClassifier:
    
    # KNN Fit Function just stores the data
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    # Euclidean Distances
    def distance(self, X, X_test):
        return np.sum((X-X_test)**2, axis = 1)**0.5
        
    # Compute the nearest class
    def predict(self, X_test, k = 3):
        all_predictions = []
        for img in X_test:
            distances = self.distance(self.X, img)
            top_k_indices = distances.argsort()[:k]
            top_k_labels = self.y[top_k_indices]
            all_labels, counts = np.unique(top_k_labels, return_counts = True)
            all_predictions.append(all_labels[counts.argmax()])
        return all_predictions

    # Compute the training accuracy
    def accuracy(self, y_true, y_pred):
        return (y_true==y_pred).mean()