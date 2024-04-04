import numpy as np
import os
import cv2
import concurrent.futures
import gc

# Set GPU acceleration flag
USE_GPU = False
FPS = 10
LENGTH = 5
PATH = 'datasets/kinetics-dataset-main/k700-2020'
# PATH = 'test-samples/'
FRAME_SIZE = (224, 224)
BATCH_SIZE = 50
INDEX = int(input('Enter index (0 to 13 inclusive): '))

# Function to read and resize videos
def process_video(video_path, gpu):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    frames = []
    for _ in range(FPS * LENGTH):
        ret, frame = cap.read()
        if not ret:
            break
        if gpu:
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
    
    return np.array([np.array(frames), video_path.split('/')[-1]])

# Function to process videos in parallel
def process_videos_parallel(video_paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        resized_videos = list(executor.map(process_video, video_paths, [USE_GPU] * len(video_paths)))
        executor.shutdown(wait=True)
        gc.collect()
    
    # create a text file to indicate that the processing is done
    if len(video_paths) > 0:
        f = open(video_paths[0].split('/')[-2] + ".txt", "w")
        f.write(video_paths[0].split('/')[-2] + " is done")
    
    return resized_videos

# Paths
train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')
val_dir = os.path.join(PATH, 'val')

# Get list of video files
train_files = [os.path.join(train_dir, folder, file) for folder in os.listdir(train_dir) for file in os.listdir(os.path.join(train_dir, folder))][INDEX * BATCH_SIZE:INDEX * BATCH_SIZE + BATCH_SIZE]
test_files = [os.path.join(test_dir, folder, file) for folder in os.listdir(test_dir) for file in os.listdir(os.path.join(test_dir, folder))][INDEX * BATCH_SIZE:INDEX * BATCH_SIZE + BATCH_SIZE]
val_files = [os.path.join(val_dir, folder, file) for folder in os.listdir(val_dir) for file in os.listdir(os.path.join(val_dir, folder))][INDEX * BATCH_SIZE:INDEX * BATCH_SIZE + BATCH_SIZE]
files = train_files + test_files + val_files

# Process videos
dataset = np.array([process_videos_parallel(files[i:i+BATCH_SIZE]) for i in range(0, len(files), BATCH_SIZE)])

# Save dataset
np.save(f'dataset_p{INDEX}.npy', dataset)

# load dataset.npy
ds = np.load(f'dataset_p{INDEX}.npy', allow_pickle=True)

print(ds.shape)

print()

# also print shape of each video
for i in range(10):
    print(ds[i][0].shape, ds[i][1])