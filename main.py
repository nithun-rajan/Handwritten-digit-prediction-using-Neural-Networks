import pandas as pd
import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
import matplotlib.pyplot as plt
import cv2 
from PIL import Image
from tensorflow import keras 
import seaborn as sns

from keras.datasets import mnist
from keras.models import Sequential
#from tensorflow.math import confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#print(X_train.shape)

plt.imshow(X_train[0]) # Display the first image in the training set
#plt.show()
#print(y_train[0]) #corresponding label

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# setting up layer 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 2D images to 1D
    keras.layers.Dense(55, activation='relu'),  # Fully connected layer with 128 neurons
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')   # Output layer with 10 neurons for 10 classes

])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train , epochs=20)
 
 #training data accuracy 99.18

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
#print('\nTest accuracy:', test_acc)

# Make predictions on the test set
predictions = model.predict(X_test)
#print(predictions[0])  # Print the predictions for the first test image

# Convert predictions to class labels
predicted_labels = [np.argmax(i) for i in predictions]
#print(predicted_labels[:10])  # Print the first 10 predicted labels

conusion_matrix = tf.math.confusion_matrix(y_test, predicted_labels)
#print(conusion_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conusion_matrix, annot=True, fmt='d', cmap='Blues')
#plt.show()


#building prediction function
input_image_path =  '/Users/nithunsundarrajan/Downloads/ok.png'

input_image = cv2.imread(input_image_path)

#cv2.imshow('Input Image', input_image)
#plt.show()

print(input_image.shape)

input_image_grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

input_image_resized = cv2.resize(input_image_grayscale, (28, 28))  # Resize to 28x28 pixels

print(input_image_resized)

input_image_normalized = input_image_resized / 255.0  # Normalize the pixel values

input_image_reshaped = input_image_normalized.reshape(1, 28, 28)  # Reshape to match model input shape