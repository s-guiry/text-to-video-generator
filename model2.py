import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv3D, concatenate, UpSampling3D, Reshape, Permute
from keras.models import Model
from tqdm import tqdm

# Setup to force use the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    device = '/gpu:0'
else:
    device = '/cpu:0'  # Falls back to CPU if no GPU is found
print(f"Using device {device}")

VIDEO_SHAPE = (224, 224)

def get_noise(shape, timestep, mean=0, base_std=1):
    # Exponential decay of noise over timesteps
    std = base_std / np.sqrt(1 + timestep)
    noise = np.random.normal(mean, std, shape)
    return noise

def noise_predictor(input_size, label_size, timestep_size):
    inputs = Input(shape=input_size)
    label_inputs = Input(shape=label_size)
    timestep_inputs = Input(shape=timestep_size)
    
    # Processing label and timestep inputs to match spatial dimensions of video inputs
    label_processed = Reshape((1, 1, 1, label_size[0]))(label_inputs)
    label_tiled = tf.tile(label_processed, [1, 50, 224, 224, 1])
    
    timestep_processed = Reshape((1, 1, 1, 1))(timestep_inputs)
    timestep_tiled = tf.tile(timestep_processed, [1, 50, 224, 224, 1])
    
    # Combining inputs
    combined_inputs = concatenate([inputs, label_tiled, timestep_tiled], axis=-1)
    
    # Defining CNN layers
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(combined_inputs)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv1)
    up1 = UpSampling3D(size=(1, 1, 1))(conv2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up1)
    output = Conv3D(3, (1, 1, 1), activation='sigmoid')(conv3)  # Assuming RGB output
    
    model = Model(inputs=[inputs, label_inputs, timestep_inputs], outputs=output)
    return model

optimizer = tf.keras.optimizers.Adam()

input_shape = (50, 224, 224, 3)
label_shape = (512,)
timestep_shape = (1,)

model = noise_predictor(input_shape, label_shape, timestep_shape)

# Load dataset
dataset = np.load('dataset_p0.npy', allow_pickle=True)
video_data = dataset[0][0]
label_data = dataset[0][1]
T = np.random.randint(1, 101)
true_noise = get_noise(video_data.shape, T)

# Training loop with explicit GPU usage
epochs = 10
with tf.device(device):  # Forces the use of GPU/CPU depending on the availability
    for epoch in tqdm(range(epochs), desc='Training Progress'):
        with tf.GradientTape() as tape:
            predictions = model([np.expand_dims(video_data, 
