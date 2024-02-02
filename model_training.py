# Classify ImageNet classes with Inception v3
# training off dataset from: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset?resource=download

# resources used
# https://www.kaggle.com/models/google/inception-v3
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3
# https://keras.io/api/applications/#usage-examples-for-image-classification-models

# tensors are a fundamental data structure representing multi-dimensional arrays, used to represent and manipulate data in neural networks

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

# instantiate inception v3 architecture
# input_shape DEFAULT value = 299,299,3
# exclude fully connected top layers to refine model but retain the base features of the model (useful general patterns)
base_model = InceptionV3(include_top=False, weights='imagenet')

x = base_model.output

# reduces spatial dimensions of the input tensor by taking the average over all values in that dimension
x = layers.GlobalAveragePooling2D()(x)

# adds a Dense layer with 256 units (neurons) and a Rectified Linear Unit activation function
# layers are neural network layers
# Dense is a fully connected layer, where each neuron is connected to every neuron in the previous layer
# common in neural network architectures for non-linearity and ability to capture complex patterns
# 256, ReLU is a common configuration
x = layers.Dense(256, activation='relu')(x)

# increase the number of units if the data is more complex
# reduce the number of units if model is overfitting (does well on trained data but not on new data)

# add final Dense layer to serve as output layer
# 3 classes = 3 output neurons
# softmax converts raw output scores into probability distributions
predictions = layers.Dense(3, activation='softmax')(x)

# creates final model
model = models.Model(inputs=base_model.input, outputs=predictions)

# freeze initial layers to:
# retain pre-trained knowledge, reduce the amount needed to be trained
for layer in base_model.layers:
    layer.trainable = False

# compile model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# image augmentation - training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# image augmentation - validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32 # numbers of images processed each iteration of training

# creates generator objects from the specified image directories
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(299, 299), # Inception v3 expects (299,299) image sizes
                                                    batch_size=batch_size, # no of images processed in each iteration of training
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(299, 299), # Inception v3 expects (299,299) image sizes
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

# Update the model.fit() function with the calculated steps_per_epoch values
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=50#,  # Adjust as needed
          #validation_data=validation_generator,
          #validation_steps=validation_generator.samples // batch_size
          )

# Save the trained model
model.save('models/covid_recognition_model.keras')

# Notes:
# 3 classes, 20-26 photos each
# 10 epochs, accurately recognized covid, mostly got normal right although it misrecognized some as viral pneumonia
# 50 epochs:
# - 18 recognized as covid, 2 recognized as normal
# - 20 recognized as normals, 
# - 14 viral pneumonia recognized correctly, 6 recognized as normal