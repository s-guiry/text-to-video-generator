import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

# Load the noise predictor model
model = tf.keras.models.load_model('noise_predictor_model.h5')

# Load Universal Sentence Encoder for prompt embedding
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

# Get prompt from user
prompt = input('Enter the prompt: ')

# Embed the prompt
embedding = embed([prompt])[0]

# Define noise generation function
def get_noise(shape, timestep, mean=0, base_std=1):
    std = base_std / (np.sqrt(1 + timestep) + 0.001)  # Avoid division by zero with small epsilon
    noise = np.random.normal(mean, std, shape)
    return noise

# Generate the initial noise
T = 100
initial_noise = get_noise((50, 224, 224, 3), T)

# Initialize video writer
video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (224, 224))

# Use tqdm to track progress
with tqdm(total=T + 1, desc='Generating Noise') as pbar:
    while T >= 0:
        # Get the predicted noise
        noise = model.predict([initial_noise, tf.reshape(embedding, (1, -1)), np.array([[T]])])
        
        # Normalize noise to [-1, 1] range
        normalized_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 2 - 1
        
        # Update initial noise
        initial_noise += 0.9 * normalized_noise
        
        # Clip noise to stay within valid image range
        initial_noise = np.clip(initial_noise, -1, 1)
        
        # Decrease the timestep
        T -= 1
        
        # Increment tqdm progress bar
        pbar.update(1)

# Scale noise back to [0, 255] range for visualization
noise_frames = ((initial_noise + 1) * 127.5).astype(np.uint8)

# Write frames to video
for frame in noise_frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Release video writer
video.release()
