import tensorflow as tf

# Assuming your tensors are TensorFlow tensors
label_inputs = tf.random.normal((512,))  # Example (512,) tensor
inputs = tf.random.normal((, 50, 224, 224, 3))  # Example (50, 224, 224, 3) tensor
timestep_inputs = tf.random.normal((1,))  # Example (1,) tensor

reshaped_label = tf.reshape(label_inputs, (1, 1, 1, 512))
reshaped_label_tiled = tf.tile(reshaped_label, [50, 224, 224, 1])

reshaped_timestep = tf.reshape(timestep_inputs, (1, 1, 1, 1))
reshaped_timestep_tiled = tf.tile(reshaped_timestep, [50, 224, 224, 1])

concatenated_input = tf.concat([reshaped_label_tiled, inputs, reshaped_timestep_tiled], axis=3)

print(concatenated_input.shape)