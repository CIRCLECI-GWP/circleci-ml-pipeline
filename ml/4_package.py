# Description:
# This script packages the trained model for deployment

# Dependencies:
# This script relies on 2_train.py having been run to generate the trained model
# In your ML pipeline, you will probably want to run this script after the test step, and only if the test passes

# Import dependencies
import tempfile
import sys
import tensorflow as tf
import os
from dotenv import load_dotenv
import pysftp

# Get the path to this script
script_dir = sys.path[0]

# Load environment variables from .env file
# This file should contain your secrets like passwords and auth keys and should be loaded from CircleCi environment variables - not stored in your repo
load_dotenv()

# Create a temporary directory
temp_dir = tempfile.TemporaryDirectory()

# Load the model from the previous step
model = tf.keras.models.load_model(os.path.join(
    script_dir, 'training_data/trained_model'))

# Read the version number from a file
# TensorFlow Serving models are versioned, and the version number is used to identify the model to use
# Version numbers are integers, and are incremented each time a new model is deployed - you will most likely manage the version number when updating your models
with open(os.path.join(script_dir, 'model_version.txt')) as f:
    version = f.readline().strip()

# Create a temporary directory to store the model
temp_export_path = os.path.join(temp_dir.name, 'model-' + version)

# Save the model
# We don't just use the existing saved model as it is likely you will want to export the model in a different format for serving, or with different parameters
tf.keras.models.save_model(
    model,
    filepath=temp_export_path,
)

# Files created in your CircleCI pipelines will not persist for the next run
# You will most likely be storing your models in a central file store
# This could be on any kind of share - SFTP, S3, etc - in this example, we're storing them on a remote server via SFTP in a folder called 'staging'
# Note that we're using a password and skipping host key checking below. You should NOT do this in production!
# You should also make sure the use your are using to run your CI/CD pipelines only has access to the remote resources required

# Save the packaged model to the remote server
# In this example, the staging location is on the same server as TensorFlow Serving deployment to keep things simple
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None  # Disable host keys checking - not for production
remote_staging_path = os.getenv('DEPLOY_SERVER_PATH') + '/staging/' + version

# Again, it pays to be verbose - any output will appear in the CircleCI web console for later inspection
print('Uploading model to: ' + remote_staging_path)

with pysftp.Connection(os.getenv('DEPLOY_SERVER_HOSTNAME'), username=os.getenv('DEPLOY_SERVER_USERNAME'), password=os.getenv('DEPLOY_SERVER_PASSWORD'), cnopts=cnopts) as sftp:
    # Make all non-existing directories
    sftp.makedirs(remote_staging_path)
    # The packaged model is a directory, so must use the put_r function to recursively upload it
    sftp.put_r(temp_export_path, remote_staging_path)

print('\nSaved model version:' + version)

# Clean up the temporary directory
temp_dir.cleanup()
