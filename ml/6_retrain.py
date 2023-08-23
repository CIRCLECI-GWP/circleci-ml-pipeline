# Description:
# This script re-trains the model

# Dependencies:
# This script relies on 4_package.py having been run to generate the packaged model and upload it to the staging location
# It's intended to retrain a previously saved model, so should be run in a full workflow, replacing the train step with this script

# Import dependencies
import tempfile
import sys
import numpy as np
import tensorflow as tf
import os
import pickle
from dotenv import load_dotenv
import pysftp

# Get the path to this script
script_dir = sys.path[0]

# Load environment variables from .env file
load_dotenv()

# Create a temporary directory to store the downloaded model
temp_dir = tempfile.TemporaryDirectory()

# Read the version number from a file
with open(os.path.join(script_dir, 'model_version.txt')) as f:
    version = f.readline().strip()

# Download the existing model from the staging location
# Note that we're using a password and skipping host key checking below. You should NOT do this in production!
remote_path = os.getenv('DEPLOY_SERVER_PATH') + '/staging/' + version
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None
with pysftp.Connection(os.getenv('DEPLOY_SERVER_HOSTNAME'), username=os.getenv('DEPLOY_SERVER_USERNAME'), password=os.getenv('DEPLOY_SERVER_PASSWORD'), cnopts=cnopts) as sftp:
    # Recursively download the directory
    sftp.get_r(remote_path, temp_dir.name)

# Reload the model downloaded model for retraining
reloaded_model = tf.keras.models.load_model(temp_dir.name + '/' + remote_path)

# Print a summary of the reloaded model
reloaded_model.summary()

# Load the data from the build step
with open(os.path.join(script_dir, 'training_data/test_images.pkl'), 'rb') as f:
    test_images = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/test_labels.pkl'), 'rb') as f:
    test_labels = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/train_images.pkl'), 'rb') as f:
    train_images = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/train_labels.pkl'), 'rb') as f:
    train_labels = pickle.load(f)

# Test the existing model before retraining
test_loss, test_acc = reloaded_model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))

# As this is an example using dummy data, we don't have any 'new' data to retrain with
# So, I'll simulate getting a bad batch of retraining data by shuffling the existing data and retraining with that
np.random.shuffle(train_images)
np.random.shuffle(train_labels)
epochs = 5
reloaded_model.fit(train_images, train_labels, epochs=epochs)

# Test the retrained model
retrain_test_loss, retrain_test_acc = reloaded_model.evaluate(
    test_images, test_labels)
print('\nTest accuracy after retrain: {}'.format(retrain_test_acc))

# If the retrained model is worse than the original, something has gone wrong (eg. we got a bad batch of retraining data)
if test_acc > retrain_test_acc:
    raise Exception(
        "Testing of retraining data failed - will not package or deploy retrained model")

# save the model for the next step
tf.keras.models.save_model(
    reloaded_model,
    filepath=os.path.join(script_dir, 'training_data/trained_model'),
)

# Clean up the temporary directory
temp_dir.cleanup()