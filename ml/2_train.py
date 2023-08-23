# Description:
# Our training phase loads the data from the build phase, trains a model, and saves it for use in the next steps

# Dependencies:
# This script relies on 1_build.py having been run to generate the training data

# Import dependencies
import sys
import tensorflow as tf
from tensorflow import keras
import os
import pickle

# Get the path to this script
script_dir = sys.path[0]

# Load the data from the build step
with open(os.path.join(script_dir, 'training_data/train_images.pkl'), 'rb') as f:
    train_images = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/train_labels.pkl'), 'rb') as f:
    train_labels = pickle.load(f)

# Train model using the simplest possible CNN as we aren't focussed on the modelling itself
model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3,
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, name='Dense')
])

# Compile the model
epochs = 5
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=epochs)

# Print a summary of the model
# It pays to be verbose when writing scripts for use in CI/CD pipelines - console output is displayed in the CircleCi web interface
# You can view it there to see the status, success, and failure of jobs, and use it to assist with debugging
model.summary()

# For larger, ongoing workflows, you could use checkpoints https://www.tensorflow.org/guide/checkpoint and monitor them from your CI/CD platform

# Save the model for the next step
tf.keras.models.save_model(
    model,
    filepath=os.path.join(script_dir, 'training_data/trained_model'),
)