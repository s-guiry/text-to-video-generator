import numpy as np
import os
import cv2
import concurrent.futures
import gc
import math
import tensorflow as tf
import tensorflow_hub as hub

# Set GPU acceleration flag
USE_GPU = True  # Assuming you want to use GPU if available for certain ops
FPS = 10
LENGTH = 5
PATH = 'datasets/kinetics-dataset-main/k700-2020'
FRAME_SIZE = (224, 224)
BATCH_SIZE = int(input(f'Enter amount of labels (1 to 700): '))

# Load Universal Sentence Encoder model from TensorFlow Hub
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    frames = []
    for _ in range(FPS * LENGTH):
        ret, frame = cap.read()
        if not ret:
            break
        if USE_GPU:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            resized_frame_gpu = cv2.cuda.resize(frame_gpu, FRAME_SIZE)
            resized_frame = resized_frame_gpu.download()
        else:
            resized_frame = cv2.resize(frame, FRAME_SIZE)
        frames.append(np.array(resized_frame, dtype=np.float32))  # Use np.float32 instead of default
    cap.release()
    
    label = video_path.split('/')[-1]
    return np.array(frames), label

def process_batch_parallel(batch):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        resized_videos = list(executor.map(process_video, batch))
    return resized_videos

def process_videos_in_batches(video_paths, batch_size):
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i + batch_size]
        yield batch

train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')
val_dir = os.path.join(PATH, 'val')

train_labels = [folder for folder in os.listdir(train_dir)]
files = [os.path.join(train_dir, folder, file) for folder in train_labels for file in os.listdir(os.path.join(train_dir, folder))]

# Limit the processing to a certain number of batches
max_batches = 100000/BATCH_SIZE  # Set this based on your memory and needs
i = 0
for batch in process_videos_in_batches(files, BATCH_SIZE):
    gc.collect()
    
    dataset = process_batch_parallel(batch)
    dataset_embedded = []
    for frames, label in dataset:
        embedding = embed([label])[0]
        dataset_embedded.append((frames, embedding))
    
    dataset_array = np.array(dataset_embedded, dtype=object)
    np.save(f'dataset_p{i}.npy', dataset_array)
    
    print(f'Done with batch {i}')
    i += 1
    if i >= max_batches:
        break
