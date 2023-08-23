# Description:
# This script tests the trained model against the test data generated in the build step

# Dependencies:
# This script relies on 2_train.py having been run to generate the trained model

# Import dependencies
import sys
import os
import pickle
import tensorflow as tf

# Get the path to this script
script_dir = sys.path[0]

# Load the model from the previous step
model = tf.keras.models.load_model(os.path.join(
    script_dir, 'training_data/trained_model'))

# Load the test data from the build step
with open(os.path.join(script_dir, 'training_data/test_images.pkl'), 'rb') as f:
    test_images = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/test_labels.pkl'), 'rb') as f:
    test_labels = pickle.load(f)

# Some very basic testing
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))

# Fail the test if it is below a certain accuracy
# The above model and test should pass, change the below value to 0.9 to see it fail
if test_acc < 0.8:
    # Raising an exception in Python will cause an error code and cause the CircleCI job to fail
    raise Exception("Test failed")
