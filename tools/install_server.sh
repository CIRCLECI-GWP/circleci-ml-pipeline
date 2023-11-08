#!/usr/bin/env bash
# If any of the steps (not in a block construction) fail, the whole script should halt in place and the script will exit with the failure message of the failed step.
set -eu -o pipefail

#Commands to spin up a quick TensorFlow Serving server for use with this example

# Install Docker
# Follow the instructions at https://docs.docker.com/engine/install/ubuntu/ for the latest official version for your platform

# Create the directories and grant permissions so that the user defined in the .env file and docker can read/write to them
sudo mkdir -p /var/models/staging # so that docker will have something to bind to, it will be populated later
sudo mkdir -p /var/models/prod
sudo chown -R $USER:docker /var/models
sudo chmod -R 775 /var/models

# To perminantely delete an existing container with the name tensorflow_serving, run this command
# docker rm -f tensorflow_serving 

# Create a TensorFlow Serving container with the directories configured for use with this example
# Ensure any previous container is removed first to avoid conflicts and startup errors
docker run -d -rm --name tensorflow_serving -p 8501:8501 -v /var/models/prod:/models/my_model -e MODEL_NAME=my_model tensorflow/serving

# Until you publish your model to TensorFlow Serving, you will receive this message: Did you forget to name your leaf directory as a number (eg. '/1/')?
# If you see this message in the Docker output, it means that the container is running successfully
