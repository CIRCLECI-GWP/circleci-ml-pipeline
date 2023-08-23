# Description:
# This script tests the deployed model on the TensorFlow Serving server

# Dependencies:
# This script relies on 5_deploy.py having been run to deploy the model to the production location
# It also requires Tenserflow Serving to be running on the production server as configured in tools/install_server.sh

# Import dependencies
import requests
import numpy as np
import json
import pickle
import os
import sys
from dotenv import load_dotenv

# Get the path to this script
script_dir = sys.path[0]

# Load environment variables from .env file
load_dotenv()

# Load the data from the build step
with open(os.path.join(script_dir, 'training_data/test_labels.pkl'), 'rb') as f:
    test_labels = pickle.load(f)

with open(os.path.join(script_dir, 'training_data/test_images.pkl'), 'rb') as f:
    test_images = pickle.load(f)

# Define the class names that the model is trained on
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Convert the test images to the format expected by the model
data = json.dumps({"signature_name": "serving_default",
                  "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

# Make the request to the TensorFlow Serving server
headers = {"content-type": "application/json"}
json_response = requests.post('http://' + os.getenv('DEPLOY_SERVER_HOSTNAME') +
                              ':8501/v1/models/my_model:predict', data=data, headers=headers)  # uses default port 8501
predictions = json.loads(json_response.text)['predictions']

# Print the results of the test
# We're not really concerned with the results here, just that the request succeeded
print('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]))
