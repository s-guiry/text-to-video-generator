import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv3D, concatenate, UpSampling3D, Reshape, Permute
from keras.models import Model
from tqdm import tqdm

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU devices found.")


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

# Define input dimensions correctly as per your data and network architecture
input_shape = (50, 224, 224, 3)
label_shape = (512,)
timestep_shape = (1,)

# Create model
model = noise_predictor(input_shape, label_shape, timestep_shape)

# Load dataset
dataset = np.load('dataset_p0.npy', allow_pickle=True)
video_data = dataset[0][0]  # Assuming video data is stored as the first element
label_data = dataset[0][1]  # Assuming label data is stored as the second element
T = np.random.randint(1, 101)  # Random timestep
true_noise = get_noise(video_data.shape, T)

# Example training loop with tqdm
epochs = 10
for epoch in tqdm(range(epochs), desc='Training Progress'):
    with tf.GradientTape() as tape:
        # Generate predictions
        predictions = model([np.expand_dims(video_data, 0), np.expand_dims(label_data, 0), np.array([[T]])])
        # Calculate loss (here using mean squared error for simplicity)
        loss = tf.reduce_mean(tf.square(predictions - true_noise))

    # Calculate gradients and update model weights
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tqdm.write(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

# Save the trained model
model.save('noise_predictor_model.h5')
print('Done training and model saved!')
