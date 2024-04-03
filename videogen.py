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

# create a np array to store the videos with their labels -- each folder is the label of all the videos in it
dataset = []

# read the train videos
def read_videos(split):
    dir = path + split + '/'
    
    for folder in dir:
        files = os.listdir(dir)
        for file in files:
            cap = cv2.VideoCapture(dir + '/' + file)
            frames = []
            for _ in range(30):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            dataset.append([np.array(frames), folder])
            cap.release()
            
# read the videos in the train, test, and val folders
read_videos('train')
read_videos('test')
read_videos('val')

# convert the dataset to a np array
dataset = np.array(dataset)
np.save('dataset.npy', dataset)