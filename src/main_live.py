#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:10:09 2021

@author: shesriva
"""

# KNN Face Recogniser from Live Stream
# run using command: python main_live.py training_data.npy training_labels.npy label_dict.json

# ------------------------------------------------------------------------------ #

import cv2
import numpy as np
import sys
from knn_classifier import KNNClassifier
import json

cap = cv2.VideoCapture(0) # param = 0 uses the integrated webcam of the system.

# Load the Face Detection Haar Cascade Classifier
cascade_classifier_path = "haarcascade_frontalface_alt.xml"
face_haar_cascade = cv2.CascadeClassifier(cascade_classifier_path) # To get the classifier object of the specified file.

X_feature_matrix = [] # Constructing the feature matrix of images
interval = 0 # Capturing an image at every `interval` step.

training_data = np.load("../training_data/" + sys.argv[1])
training_labels = np.load("../training_data/" + sys.argv[2])
# label_dict = json.load(sys.argv[3])

with open("../training_data/" + sys.argv[3], 'r') as fp:
    label_dict = json.load(fp)

knn = KNNClassifier()
knn.fit(training_data, training_labels)

# Keep Capturing Images from Webcam
while True:
    
    # reading the image
    # returned_value : if the camera is accessible and an image has been captured.
    # returned_frame : the returned image from webcam
    returned_value, returned_frame = cap.read()
    
    # print(type(returned_frame)) # The returned image is a <class 'numpy.ndarray'>
    
    # Capture only the face of the image
    # scaleFactor --> to reduce the image by a factor of (scaleFactor-1) * 100 %
    # for ex: scaleFactor = 1.3 reduces the image by (((1.3-1) = 0.3) * 100) = 30%. 
    faces = face_haar_cascade.detectMultiScale(returned_frame, scaleFactor = 1.3) # Detects all the faces in the captured frame.
    # print(type(faces)) # Again, these are numpy ndarrays with size 4*1 containing (x_top_left, y_top_left, width, height) of every detected face.
    
    print("CheckPoint1")
    
    padding = 100
    for face in faces:
        x_top_left, y_top_left, width, height = face # Fetching the 4 coordinates for every face.

        # Capture every 10 seconds.
        if interval % 10 == 0:
            # THE X and Y are interchanged in the returned_frame. TAKE CARE OF THIS.
            face_image = returned_frame[y_top_left - padding : y_top_left + height + padding,
                                    x_top_left - padding : x_top_left + width + padding]
            face_image = cv2.resize(face_image, (150, 150)) # Standardize the image size.
            
        interval += 1

    print("CheckPoint2")
    
    # Show the captured image with a rectangle around the face.
    cv2.imshow("Captured Frame: ", returned_frame)
    cv2.imshow('Captured Face:', face_image)
    
    # Wait for 1ms for a key to be pressed. If a key is pressed it returns the 
    # ASCII value of the key pressed otherwise it returns -1.
    # If a key is pressed and it equals 'q' here, the loop breaks.
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        print(chr(key_pressed), key_pressed) # print the character value. key_pressed gives ascii value.
        break
        

# Releases the webcam held through VideoCapture object.
cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------ #
