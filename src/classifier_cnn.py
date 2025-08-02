# classifier_cnn.py
# Author: Julia Johnson
# Project: Image Classification with CNN - CIFAR-10 Dataset
# Description: Builds and trains a Convolutional Neural Network (CNN) to classify CIFAR-10 images.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

print(f"Train Images Shape: {train_images.shape}")
print(f"Test Images Shape: {test_images.shape}")

# Step 2: Build the CNN model
print("Building the CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Step 3: Compile the model
print("Compiling the model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Step 4: Train the model
print("Training the model...")
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Step 5: Evaluate the model
print("Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}')

# Step 6: Visualize training results
print("Plotting training results...")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1])
plt.show()
