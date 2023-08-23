#!/bin/bash

# Runs the retrain workflow for testing purposes

# Exit when any command fails
set -e

python3 ./ml/1_build.py 
python3 ./ml/6_retrain.py
python3 ./ml/4_package.py
python3 ./ml/5_deploy.py
sleep 5 # Wait for the model to be deployed
python3 ./ml/7_test_deployed_model.py
echo "Success!"