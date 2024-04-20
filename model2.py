import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv3D, concatenate, UpSampling3D, Reshape, Permute, BatchNormalization
from keras.models import Model
from tqdm import tqdm
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('float32')


# Setup for mixed precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

VIDEO_SHAPE = (224, 224)

def get_noise(shape, timestep, mean=0, base_std=1):
    std = base_std / (np.sqrt(1 + timestep) + 0.001)  # Adding a small epsilon to avoid division by zero
    noise = np.random.normal(mean, std, shape)
    return noise

def noise_predictor(input_size, label_size, timestep_size):
    inputs = Input(shape=input_size)
    label_inputs = Input(shape=label_size)
    timestep_inputs = Input(shape=timestep_size)
    
    # Efficient processing of label and timestep inputs
    label_broadcasted = Reshape((1, 1, 1, label_size[0]))(label_inputs)
    label_broadcasted = tf.broadcast_to(label_broadcasted, [1, 50, 224, 224, label_size[0]])
    
    timestep_broadcasted = Reshape((1, 1, 1, 1))(timestep_inputs)
    timestep_broadcasted = tf.broadcast_to(timestep_broadcasted, [1, 50, 224, 224, 1])
    
    combined_inputs = concatenate([inputs, label_broadcasted, timestep_broadcasted], axis=-1)
    
    # CNN layers with optimization
    conv1 = Conv3D(64, (3, 3, 3), padding='same', use_bias=False)(combined_inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = tf.nn.relu(conv1)
    conv2 = Conv3D(128, (3, 3, 3), padding='same', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = tf.nn.relu(conv2)
    up1 = UpSampling3D(size=(1, 1, 1))(conv2)
    conv3 = Conv3D(64, (3, 3, 3), padding='same', use_bias=False)(up1)
    conv3 = BatchNormalization()(conv3)
    conv3 = tf.nn.relu(conv3)
    output = Conv3D(3, (1, 1, 1), activation='sigmoid')(conv3)
    
    model = Model(inputs=[inputs, label_inputs, timestep_inputs], outputs=output)
    return model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

input_shape = (50, 224, 224, 3)
label_shape = (512,)
timestep_shape = (1,)

model = noise_predictor(input_shape, label_shape, timestep_shape)

dataset = np.load('dataset_p0.npy', allow_pickle=True)
video_data = dataset[0][0]
label_data = dataset[0][1]
T = np.random.randint(1, 101)
true_noise = get_noise(video_data.shape, T)

epochs = 10
for epoch in tqdm(range(epochs), desc='Training Progress'):
    with tf.GradientTape() as tape:
        predictions = model([np.expand_dims(video_data, 0), np.expand_dims(label_data, 0), np.array([[T]])])
        loss = tf.reduce_mean(tf.square(predictions - true_noise))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tqdm.write(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
    

# Save the trained model
model.save('noise_predictor_model.h5')
print('Done training and model saved!')
