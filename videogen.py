import numpy as np
import os
import cv2

# replace this with the actual data path
#path = 'datasets/kinetics-dataset-main/k700-2020/'
path = 'test-samples/'

# get the list of files in the train, test, and val folders
train_files = os.listdir(path + 'train')
test_files = os.listdir(path + 'test')
val_files = os.listdir(path + 'val')

# create a np array to store the videos
dataset = np.zeros((len(train_files) + len(test_files) + len(val_files), 90, 128, 128, 3))

# function to add frames to the dataset
def add_frames(video, i_start, path):
    for i, file in enumerate(train_files):
        video = cv2.VideoCapture(path + file)
        frames = []
        for _ in range(30):
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame)
        dataset[i + i_start] = np.array(frames)

# add frames to the dataset for train, test, and val
add_frames(dataset, 0, path + 'train/')
add_frames(dataset, len(train_files), path + 'test/')
add_frames(dataset, len(train_files) + len(test_files), path + 'val/')

# save the dataset to a file
np.save('dataset.npy', dataset)