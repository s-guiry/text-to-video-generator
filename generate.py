import tensorflow as tf
import cv2

# load the model
model = tf.keras.models.load_model('noise_predictor_model.h5')

prompt = input('What would you like to generate? ')

# generate the output
output = model(prompt)

video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (224, 224))
for i in range(50):
    data = output[i].numpy()
    video.write(data)
video.release()