import numpy as np
import os
import cv2
import concurrent.futures
import gc
import math
import tensorflow as tf
import tensorflow_hub as hub

# Set GPU acceleration flag
USE_GPU = False
FPS = 10
LENGTH = 5
PATH = 'datasets/kinetics-dataset-main/k700-2020'
# PATH = 'test-samples/'
FRAME_SIZE = (224, 224)
BATCH_SIZE = int(input(f'Enter batch size (1 to 700): '))

# Load Universal Sentence Encoder model from TensorFlow Hub
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

# Function to read and resize videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    frames = []
    for _ in range(FPS * LENGTH):
        ret, frame = cap.read()
        if not ret:
            break
        if USE_GPU:
            # Convert frame to appropriate format for GPU operations
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            # Resize frame on GPU
            resized_frame_gpu = cv2.cuda.resize(frame_gpu, FRAME_SIZE)
            # Download resized frame from GPU
            resized_frame = resized_frame_gpu.download()
        else:
            # CPU-based resizing
            resized_frame = cv2.resize(frame, FRAME_SIZE)
        # append the frame as well as its label (folder name)
        frames.append(np.array(resized_frame))
    cap.release()
    
    if np.random.randint(1, 501) == 20:
        print('Still going!', flush=True)
    
    label = video_path.split('/')[-1]

    return np.array(frames), label

# Function to process videos in parallel for a single batch
def process_batch_parallel(batch):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        resized_videos = list(executor.map(process_video, batch))
    return resized_videos

# Function to process videos in batches
def process_videos_in_batches(video_paths, batch_size):
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i + batch_size]
        yield batch

# Paths
train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')
val_dir = os.path.join(PATH, 'val')

# get only the first BATCH_SIZE labels from INDEX
train_labels = [folder for folder in os.listdir(train_dir)]
test_labels = [folder for folder in os.listdir(test_dir)]
val_labels = [folder for folder in os.listdir(val_dir)]

# Get list of video files using the labels
train_files = [os.path.join(train_dir, folder, file) for folder in train_labels for file in os.listdir(os.path.join(train_dir, folder))]
test_files = [os.path.join(test_dir, folder, file) for folder in test_labels for file in os.listdir(os.path.join(test_dir, folder))]
val_files = [os.path.join(val_dir, folder, file) for folder in val_labels for file in os.listdir(os.path.join(val_dir, folder))]
files = train_files + test_files + val_files
    
# Process videos in batches
i = 0
for batch in process_videos_in_batches(files, BATCH_SIZE):
    gc.collect()
    
    dataset = process_batch_parallel(batch)
    dataset_embedded = []
    for frames, label in dataset:
        # Embed the label using Universal Sentence Encoder
        embedding = embed([label])[0]
        dataset_embedded.append((frames, embedding))
    
    dataset_array = np.array(dataset_embedded, dtype=object)

    # Save dataset
    np.save(f'dataset_p{i}.npy', dataset_array)
    
    print(f'Done with index {i}')
    print()

    i += 1

    if i * BATCH_SIZE >= 3:
        break
