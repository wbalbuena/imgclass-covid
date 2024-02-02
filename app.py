# Classify ImageNet classes with Inception v3

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import os

model_path = 'models/covid_recognition_model.keras'

# instantiate inception v3 architecture
#model = InceptionV3(weights='imagenet')

# load fine-tuned model
model = tf.keras.models.load_model(model_path)

folder_path = 'dataset/test/Covid'
#folder_path = 'dataset/test/Normal'
#folder_path = 'dataset/test/Viral Pneumonia'

images = [f for f in os.listdir(folder_path)]

for img_file in images:
    img_path = os.path.join(folder_path, img_file)

    #load image into PIL format
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)

    #add new dimension to array at the beginning
    #treat image as a batch
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)

    # Get the index of the predicted class
    predicted_class = np.argmax(preds, axis=1)[0]

    #print(img_path, ' predicted:', decode_predictions(preds, top=3)[0])
    print(img_path, ' predicted class index:', predicted_class)

print('\nDone')