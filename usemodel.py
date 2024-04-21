import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

# Load the model and embedding module
model = tf.keras.models.load_model('noise_predictor_model.h5')
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

# Get user input and compute embedding
prompt = input('Enter the prompt: ')
embedding = embed([prompt])[0]

# Define the noise generation function
def get_noise(shape, timestep, mean=0, base_std=1):
    std = base_std / (np.sqrt(1 + timestep) + 0.001)
    noise = np.random.normal(mean, std, shape)
    return noise

# Setup initial noise and replication
T = 100
initial_noise = get_noise((1, 50, 224, 224, 3), T)
replicated_embedding = tf.tile(tf.reshape(embedding, (1, -1)), [1, 1])
replicated_timestep = tf.tile(tf.reshape(np.array([T]), (1, 1)), [1, 1])

# Create a video writer object
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (224, 224))
with tqdm(total=T + 1, desc='Generating Noise') as pbar:
    previous_noise = np.copy(initial_noise)  # To store previous state
    for i in range(T + 1):
        # Generate noise and normalize
        noise = model.predict([initial_noise, replicated_embedding, replicated_timestep])
        normalized_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 2 - 1
        
        # Update initial noise with a conservative approach
        if i > 0:  # Update using a blend of previous and new noise
            initial_noise = 0.1 * previous_noise + 0.9 * normalized_noise
        else:
            initial_noise = normalized_noise
        
        # Apply a soft normalization to keep some dynamic range
        max_abs_ini_noise = np.max(np.abs(initial_noise))
        initial_noise /= max_abs_ini_noise
        
        previous_noise = np.copy(initial_noise)  # Update the previous noise

        # Prepare and write the frame
        frame = ((initial_noise[0, 0] + 1) * 127.5).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
        pbar.update(1)

video.release()
