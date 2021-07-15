# face-recognition-knn

This code repository holds python files for simple Face Recognition using KNN.

Please create the following directories at the root level of the repository before running:

1. raw_data/
2. training_data/
3. test_data/

## data_collection.py 

This script lets you use your webcam to pick frames at every 2 seconds and capture every face in the frame using HAAR-Cascase features. These frames are then stored as images into a numpy array. 

Run script using command on terminal: python data_collection.py personname.npy

For Example: Let the name of the Person be SHEFALI, the command: python data_collection.py shefali.npy 

This will create the file named shefali.npy in raw_data/ folder. 

This code can be run individually for every person to generate data. 

## construct_data_matrices.npy

This scripts combines the data generated in the raw_data/ folder and splits into training and test data sets after shuffling. It stores the training data in the folder training_data/ and the test data in the folder test_data/ 

Run script using command on terminal: python construct_data_matrices.py training_data.npy training_labels.npy label_dict.json test_data.npy test_labels.npy

This will create files named training_data.npy, training_labels.npy and label_dict.json in the training_data/ folder and test_data.npy, test_labels.npy in the test_data/ folder.

## main.py

This script loads data from test set and generates predictions. It also tests for accuracy from the test_labels. 

Run script using command on terminal: python main.py training_data.npy training_labels.npy label_dict.json test_data.npy test_labels.npy
