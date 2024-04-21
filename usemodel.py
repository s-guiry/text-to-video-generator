import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

# Load resources
model = tf.keras.models.load_model('noise_predictor_model.h5')
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
prompt = input('Enter the prompt: ')
embedding = embed([prompt])[0]

# Noise generation
def get_noise(shape, timestep, mean=0, base_std=1):
    std = base_std / (np.sqrt(1 + timestep) + 0.001)
    noise = np.random.normal(mean, std, shape)
    return noise

# Initialize variables
T = 100
initial_noise = get_noise((1, 50, 224, 224, 3), T)

# Prepare video writer
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (224, 224))
with tqdm(total=T + 1, desc='Generating Noise') as pbar:
    for _ in range(T + 1):
        noise = model.predict([initial_noise, tf.tile(tf.reshape(embedding, (1, -1)), [1, 1]), tf.tile(tf.reshape(np.array([T]), (1, 1)), [1, 1])])
        print(f"Noise Stats - Min: {np.min(noise)}, Max: {np.max(noise)}, Mean: {np.mean(noise)}")

        # Adaptive normalization
        noise_range = np.max(noise) - np.min(noise)
        normalized_noise = (noise - np.min(noise)) / noise_range if noise_range != 0 else noise
        frame = (normalized_noise[0, 0] * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Preview", frame_bgr)
        cv2.waitKey(10)

        video.write(frame_bgr)
        pbar.update(1)

video.release()
cv2.destroyAllWindows()
