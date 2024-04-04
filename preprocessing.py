# import numpy as np
# import os
# import cv2

# # replace this with the actual data path
# #path = 'datasets/kinetics-dataset-main/k700-2020/'
# path = 'test-samples/'
# fps = 10

# # get the list of files in the train, test, and val folders
# train_files = os.listdir(path + 'train')
# test_files = os.listdir(path + 'test')
# val_files = os.listdir(path + 'val')

# # create a np array to store the videos with their labels -- each folder is the label of all the videos in it
# dataset = []

# # read the train videos
# def read_videos(split):
#     dir = path + split + '/' # path as text
    
#     for folder in os.listdir(dir):
#         folder_path = dir + folder + '/'
        
#         for file in os.listdir(folder_path):
#             cap = cv2.VideoCapture(folder_path + file)
#             cap.set(cv2.CAP_PROP_FPS, fps)
#             frames = []
#             for _ in range(fps * 5):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frames.append(cv2.resize(frame, (128, 128)))
#             dataset.append(np.array([np.array(frames), folder]))
#             cap.release()
    
#     f = open(split + ".txt", "w")
#     f.write(split + " is done")
#     f.close()
            
# read_videos('train')
# read_videos('test')
# read_videos('val')

# # convert the dataset to a np array
# dataset = np.array(dataset)
# np.save('dataset.npy', dataset)

# # load dataset.npy
# ds = np.load('dataset.npy', allow_pickle=True)

# print(ds.shape)

# print()

# # also print shape of each video
# for i in range(10):
#     print(ds[i][0].shape, ds[i][1])


import numpy as np
import os
import cv2
import concurrent.futures

# Set GPU acceleration flag
USE_GPU = False

# Function to read and resize videos
def process_video(video_path, gpu):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, fps)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if gpu:
            # Convert frame to appropriate format for GPU operations
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            # Resize frame on GPU
            resized_frame_gpu = cv2.cuda.resize(frame_gpu, (128, 128))
            # Download resized frame from GPU
            resized_frame = resized_frame_gpu.download()
        else:
            # CPU-based resizing
            resized_frame = cv2.resize(frame, (128, 128))
        frames.append(resized_frame)
    cap.release()
    
    if np.random.randint(1, 21) == 20:
        print(f'Still going!')
    
    return np.array(frames)

# Function to process videos in parallel
def process_videos_parallel(video_paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        resized_videos = list(executor.map(process_video, video_paths, [USE_GPU] * len(video_paths)))
    return resized_videos

# Paths
path = 'test-samples/'
fps = 10
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')
val_dir = os.path.join(path, 'val')

# Get list of video files
train_files = [os.path.join(train_dir, folder, file) for folder in os.listdir(train_dir) for file in os.listdir(os.path.join(train_dir, folder))]
test_files = [os.path.join(test_dir, folder, file) for folder in os.listdir(test_dir) for file in os.listdir(os.path.join(test_dir, folder))]
val_files = [os.path.join(val_dir, folder, file) for folder in os.listdir(val_dir) for file in os.listdir(os.path.join(val_dir, folder))]

# Process videos
resized = process_videos_parallel(train_files + test_files + val_files)

# Combine datasets
dataset = np.array(resized)

# Save dataset
np.save('dataset.npy', dataset)