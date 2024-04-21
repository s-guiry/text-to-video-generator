import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

model = tf.keras.models.load_model('noise_predictor_model.h5')
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
prompt = input('Enter the prompt: ')
embedding = embed([prompt])[0]

def get_noise(shape, timestep, mean=0, base_std=1):
    std = base_std / (np.sqrt(1 + timestep) + 0.001)
    noise = np.random.normal(mean, std, shape)
    return noise

T = 100
initial_noise = get_noise((1, 50, 224, 224, 3), T)

replicated_embedding = tf.tile(tf.reshape(embedding, (1, -1)), [1, 1])
replicated_timestep = tf.tile(tf.reshape(np.array([T]), (1, 1)), [1, 1])

video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (224, 224))
with tqdm(total=T + 1, desc='Generating Noise') as pbar:
    for i in range(T + 1):
        noise = model.predict([initial_noise, replicated_embedding, replicated_timestep])
        normalized_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 2 - 1
        print(f"Normalized Noise Stats - Iteration {i}: Min: {np.min(normalized_noise)}, Max: {np.max(normalized_noise)}, Mean: {np.mean(normalized_noise)}")

        initial_noise += normalized_noise  # Updated rule without scaling to check impact
        print(f"Initial Noise Stats - Iteration {i}: Min: {np.min(initial_noise)}, Max: {np.max(initial_noise)}, Mean: {np.mean(initial_noise)}")

        frame = ((initial_noise[0, 0] + 1) * 127.5).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)  # Write frame to video
        
        pbar.update(1)

video.release()
