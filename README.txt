Detecting Drowsiness Level in Video Data

Team Members:
1. Srinivas Baskar
2. Jitendra Marndi

The files below are as follows:

1. frame_generator.py : extracts frames from the video data and saves it to a location.
2. landmarks_data.py : reads the above generated frames and extracts the landmarks from it and saves it to a txt file.
3. feature_extraction.py : uses the landmarks and extracts features for training from it. Saves it to a text file.
4. load_data_features.py : loads the features from the above text file into batches of 5 for training.
5. drowsiness_detection_features.py : trains the model.
6. test_model.py : loads the saved model and runs it on test data.
7. application.py : used for real time drowsiness detection.
8. utility.py : helper class used by landmarks_data.py.

Once, the video data is downloaded, run the files in the above given order. The path to the dataset should be ../data/. Change it in the frame_generator.py if needed.

Dataset: https://sites.google.com/view/utarldd/home