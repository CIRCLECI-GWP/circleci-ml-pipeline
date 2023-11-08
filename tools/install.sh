#!/usr/bin/env bash
# If any of the steps (not in a block construction) fail, the whole script should halt in place and the script will exit with the failure message of the failed step.
set -eu -o pipefail
# This script sets up a Python virtual environment and installs the required packages for running the ML workflow scripts in the ml directory
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt # Note that these requirements were built on Ubuntu 22.04
