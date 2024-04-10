import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv3D, concatenate, UpSampling3D
from keras.layers import Reshape, Permute
import threading
from tqdm import tqdm

VIDEO_SHAPE = (224, 224)

def get_noise(shape, timestep, mean=0, base_std=1):
    std = base_std * np.sqrt(timestep)
    noise = np.random.normal(mean, std, shape)
    return noise

def noise_predictor(input_size, label_size, timestep_size):
    inputs = Input(input_size)
    label_inputs = Input(label_size)
    timestep_inputs = Input(timestep_size)
    
    label_reshaped = Reshape((1, 1, label_size[0], 1))(label_inputs)
    label_permuted = Permute((4, 1, 2, 3))(label_reshaped)
    label_tiled = tf.tile(label_permuted, [1, 50, input_size[1], input_size[2], 1])
    
    timestep_reshaped = Reshape((1, 1, timestep_size[0], 1))(timestep_inputs)
    timestep_permuted = Permute((4, 1, 2, 3))(timestep_reshaped)
    timestep_tiled = tf.tile(timestep_permuted, [1, 50, input_size[1], input_size[2], 1])
    
    concatenated_input = concatenate([inputs, label_tiled, timestep_tiled], axis=-1)

    conv1 = Conv3D(64, 3, activation='relu', padding='same')(concatenated_input)
    conv2 = Conv3D(128, 3, activation='relu', padding='same')(conv1)
    
    up1 = UpSampling3D(size=(1, 1, 1))(conv2)
    conv3 = Conv3D(64, 3, activation='relu', padding='same')(up1)
    
    output = Conv3D(3, 1, activation='sigmoid')(conv3)
    
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
    denoised_video = tf.expand_dims(noisy_video, axis=0)
    predicted_noise = tf.zeros_like(denoised_video[0])
    label_data = tf.expand_dims(tf.convert_to_tensor(label_data), axis=0)
    timestep = tf.expand_dims(tf.convert_to_tensor(timestep), axis=0)

    for _ in range(num_iterations):
        predicted_noise_label = noise_predictor_model([denoised_video, label_data, timestep])
        predicted_noise = noise_predictor_model([denoised_video, np.zeros_like(label_data), timestep])
        predicted_noise = np.abs((predicted_noise - predicted_noise_label) * 2.0)
        
        iteration_loss = tf.keras.losses.mean_squared_error(predicted_noise, true_noise)
        total_loss += tf.reduce_mean(iteration_loss)
        denoised_video = subtract_noise(denoised_video, predicted_noise)
        denoised_video = reintegrate_noise(denoised_video, predicted_noise, reintegration_factor)

    return total_loss

input_shape = (50, *VIDEO_SHAPE, 3)
label_shape = (512,)
timestep_shape = (1,)
noise_predictor_model = noise_predictor(input_shape, label_shape, timestep_shape)
optimizer = tf.keras.optimizers.Adam()

num_epochs = 100
reintegration_factor = 0.9

<<<<<<< HEAD
def load_dataset(dataset, idx):
    dataset[idx] = np.load(f'dataset_p{idx}.npy', allow_pickle=True)

for epoch in range(num_epochs):
    print(f'On epoch {epoch}', flush=True)
    
    datasets = [None] * 100
    threads = [threading.Thread(target=load_dataset, args=(datasets, i)) for i in range(100)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    with tqdm(datasets, desc=f'Epoch {epoch}', unit='dataset') as pbar:
        for dataset in pbar:
            if dataset is None:
                continue
            
            for batch in dataset:
                video_data = batch[0]
                label_data = batch[1]
                
                T = np.random.randint(1, 101)
                true_noise = get_noise(video_data.shape, timestep=T)
                
                with tf.GradientTape() as tape:
                    loss = iterative_diffusion_loss(true_noise, video_data + true_noise, T, label_data, noise_predictor_model, reintegration_factor, num_iterations=100)
                
                gradients = tape.gradient(loss, noise_predictor_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, noise_predictor_model.trainable_variables))
                
                # Update progress bar description
                pbar.set_description(f'Epoch {epoch} - Loss: {loss.numpy():.4f}')
                
                if np.random.randint(10) == 5:
                    pbar.write('Still going!')
=======
# for epoch in range(num_epochs):
#     print(f'On epoch {epoch}', flush=True)
    
#     for i in range(100):
#         print(f'On epoch {i}', flush=True)
        
#         dataset = np.load(f'dataset_p{i}.npy', allow_pickle=True)
        
#         c = 0
#         for batch in dataset:
#             video_data = batch[0]
#             label_data = batch[1]
            
#             print('got labels', flush=True)
            
#             T = np.random.randint(1, 101)
#             true_noise = get_noise(video_data.shape, timestep=T)  # Generate true noise
            
#             print('generated noise', flush=True)
            
#             with tf.GradientTape() as tape:
#                 loss = iterative_diffusion_loss(true_noise, video_data + true_noise, T, label_data, noise_predictor_model, reintegration_factor, num_iterations=100)
            
#             gradients = tape.gradient(loss, noise_predictor_model.trainable_variables)
#             optimizer.apply_gradients(zip(gradients, noise_predictor_model.trainable_variables))
            
#             if c % 10 == 0:
#                 print(f'On batch {batch}', flush=True)
                
#             c += 1
            
#             break # temp
        
#         break # temp
                        
#     print(f'Epoch {epoch} loss: {loss}\n', flush=True)

dataset = np.load(f'dataset_p0.npy', allow_pickle=True)
print('loaded dataset', flush=True)
video_data = dataset[0][0]
print('got video data', flush=True)
label_data = dataset[0][1]
print('got label', flush=True)
T = np.random.randint(1, 101)
true_noise = get_noise(video_data.shape, timestep=T)  # Generate true noise
print('generated noise', flush=True)

with tf.GradientTape() as tape:
    print('time for loss', flush=True)
    loss = iterative_diffusion_loss(true_noise, video_data + true_noise, T, label_data, noise_predictor_model, reintegration_factor, num_iterations=100)
print('get gradients', flush=True)
gradients = tape.gradient(loss, noise_predictor_model.trainable_variables)
print('apply gradients', flush=True)
optimizer.apply_gradients(zip(gradients, noise_predictor_model.trainable_variables))

noise_predictor_model.save('noise_predictor_model.h5')
print('Done training!', flush=True)
>>>>>>> 1f644481c0841ad88ec8704b391b6f21c46ddb93
