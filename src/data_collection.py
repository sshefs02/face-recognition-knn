#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:18:01 2021

@author: shesriva
"""

# ------------------------------------------------------------------------------ #

# We are creating a dataset of faces from the webcam.
# run using command: python data_collection.py personname.npy
# the file gets saved to the location ../raw_data/filename.npy

# ------------------------------------------------------------------------------ #

# Every Colored Image pixel is stored in 1 byte == 8 bits. That is why
# its intensity ranges from 0 to 255.
# Every pixel has intensity in R, G and B. So essentially every pixel has 3 bytes.
# OpenCV Reads images in BGR format and not RGB.

# ------------------------------------------------------------------------------ #

import cv2
import numpy as np
import sys
import os

cap = cv2.VideoCapture(0) # param = 0 uses the integrated webcam of the system.

# Load the Face Detection Haar Cascade Classifier
face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") # To get the classifier object of the specified file.

X_feature_matrix = [] # Constructing the feature matrix of images
interval = 0 # Capturing an image at every `interval` step.

print(sys.argv)

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
    
    padding = 100
    for face in faces:
        x_top_left, y_top_left, width, height = face # Fetching the 4 coordinates for every face.
        # cv2.rectangle(returned_frame, 
        #               (x_top_left, y_top_left), 
        #               (x_top_left + width, y_top_left + height),
        #               (0, 0, 255), 2) # Top left and Bottom right points along with the frame. X increases to the left, y increases downwards.
        
        # Capture every 2 seconds.
        if interval % 2 == 0:
            # THE X and Y are interchanged in the returned_frame. TAKE CARE OF THIS.
            face_image = returned_frame[y_top_left - padding : y_top_left + height + padding,
                                    x_top_left - padding : x_top_left + width + padding]
            face_image = cv2.resize(face_image, (150, 150)) # Standardize the image size.
            # print(face_image.shape) # (x, y, channels)
            X_feature_matrix.append(face_image.flatten()) # Flatten to 150*150*3 size for every image.
        
        interval += 1

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
        
# Convert the Feature Matrix into a numpy array and save it into the filename provided as argument while running. 
X_feature_matrix = np.array(X_feature_matrix)
print(X_feature_matrix.shape)
filename = sys.argv[1] # Fetch the argument 1 as argument 0 is the code file .py
file_path = '../raw_data/' + filename
np.save(file_path, X_feature_matrix)

# Releases the webcam held through VideoCapture object.
cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------ #
