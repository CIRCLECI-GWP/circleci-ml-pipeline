#!/usr/bin/env bash
# If any of the steps (not in a block construction) fail, the whole script should halt in place and the script will exit with the failure message of the failed step.
set -eu -o pipefail

# Runs the build workflow steps for testing purposes

# Exit when any command fails
set -e

python3 ./ml/1_build.py
python3 ./ml/2_train.py
python3 ./ml/3_test.py
python3 ./ml/4_package.py
python3 ./ml/5_deploy.py
sleep 5 # Wait for the model to be deployed
python3 ./ml/7_test_deployed_model.py
echo "Success!"
