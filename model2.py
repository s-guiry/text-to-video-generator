import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv3D, concatenate, UpSampling3D

VIDEO_SHAPE = (224, 224)

def get_noise(shape, timestep, mean=0, base_std=1):
    std = base_std * np.sqrt(timestep)
    noise = np.random.normal(mean, std, shape)
    return noise

def noise_predictor(input_shape, label_shape, timestep_shape):
    # Define input layers
    inputs = Input(input_shape)
    label_inputs = Input(label_shape)
    timestep_inputs = Input(timestep_shape)

    label_reshaped = tf.reshape(label_inputs, (1, 1, 512))
    timestep_reshaped = tf.expand_dims(tf.expand_dims(timestep_inputs, axis=0), axis=0)

    # Concatenate inputs along axis -1
    concatenated_input = tf.concat([inputs, label_reshaped, timestep_reshaped], axis=-1)

    # Define convolutional layers
    conv1 = Conv3D(64, 3, activation='relu', padding='same')(concatenated_input)
    conv2 = Conv3D(128, 3, activation='relu', padding='same')(conv1)
    
    # Upsampling and additional convolutional layer
    up1 = UpSampling3D(size=(2, 2, 2))(conv2)
    conv3 = Conv3D(64, 3, activation='relu', padding='same')(up1)
    
    # Output layer
    output = Conv3D(1, 1, activation='sigmoid')(conv3)

    # Define and return the model
    model = tf.keras.Model(inputs=[inputs, label_inputs, timestep_inputs], outputs=output)
    return model

def subtract_noise(video_data, predicted_noise):
    denoised_video = video_data - predicted_noise
    return denoised_video

def reintegrate_noise(denoised_video, predicted_noise, reintegration_factor):
    reintegrated_video = denoised_video + reintegration_factor * predicted_noise
    return reintegrated_video

def iterative_diffusion_loss(true_noise, noisy_video, timestep, label_data, noise_predictor_model, reintegration_factor, num_iterations):
    total_loss = 0.0
    denoised_video = noisy_video  # Initialize denoised video with noisy video
    predicted_noise = tf.zeros_like(denoised_video[0])  # Initial noise prediction

    for _ in range(num_iterations):
        input_data_label = tf.concat([denoised_video, tf.expand_dims(label_data, axis=0), tf.expand_dims(tf.constant(timestep, dtype=tf.float32), axis=0)], axis=-1)
        predicted_noise_label = noise_predictor_model(input_data_label)
        
        input_data = tf.concat([denoised_video, tf.expand_dims(np.zeros_like(label_data), axis=0), tf.expand_dims(tf.constant(timestep, dtype=tf.float32), axis=0)], axis=-1)
        predicted_noise = noise_predictor_model(input_data)
        
        # take the difference between the predictions and amplify the noise
        predicted_noise = (predicted_noise - predicted_noise_label) * 2.0
        
        iteration_loss = tf.keras.losses.mean_squared_error(predicted_noise, true_noise)
        total_loss += tf.reduce_mean(iteration_loss)
        denoised_video = subtract_noise(denoised_video, predicted_noise)
        denoised_video = reintegrate_noise(denoised_video, predicted_noise, reintegration_factor)

    return total_loss

# Define model and optimizer
input_shape = (50, *VIDEO_SHAPE, 3)  # Assuming RGB channels
label_shape = (512,)  # Assuming the label shape
timestep_shape = (1,)  # Assuming the timestep shape
noise_predictor_model = noise_predictor(input_shape, label_shape, timestep_shape)
optimizer = tf.keras.optimizers.Adam()

# Train the model
num_epochs = 100
reintegration_factor = 0.9
for epoch in range(num_epochs):
    print(f'On epoch {epoch}')
    
    for i in range(100):
        print(i)
        
        dataset = np.load(f'dataset_p{i}.npy')
        
        for batch in dataset:
            video_data = batch[0]
            label_data = batch[1]
            true_noise = get_noise(video_data.shape, timestep=np.random.randint(1, 101))  # Generate true noise
            
            with tf.GradientTape() as tape:
                loss = iterative_diffusion_loss(true_noise, video_data + true_noise, epoch, label_data, noise_predictor_model, reintegration_factor, num_iterations=100)
            
            gradients = tape.gradient(loss, noise_predictor_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, noise_predictor_model.trainable_variables))
            
    print(f'Epoch {epoch} loss: {loss}\n')
        
noise_predictor_model.save('noise_predictor_model.h5')
print('Done training!')
