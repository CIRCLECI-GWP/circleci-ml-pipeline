# Description:
# This script deploys the trained model to a remote server
# Deployment is done as a separate step to packaging so that a history of packaged models can be kept and for greater control of the ML workflow

# Dependencies:
# This script relies on 4_package.py having been run to generate the packaged model

# Import dependencies
from dotenv import load_dotenv
import paramiko
import os
import sys

# Get the path to this script
script_dir = sys.path[0]

# Load environment variables from .env file
load_dotenv()

# Read the version number from a file
with open(os.path.join(script_dir, 'model_version.txt')) as f:
    version = f.readline().strip()

# To deploy, we'll copy the most recent packaged model from the staging location to the production location
remote_staging_path = os.getenv(
    'DEPLOY_SERVER_PATH') + '/staging/' + str(version)

# Define the Bash commands that will perform this copy on the remove server via SSH
commands = (
    # Stop the docker container by its name - if it is running
    # Tensorflow Serving should happily reload the model even without being stopped while it is swapped out, but this is an example of some of the tasks you may want to do in your own ML workflows
    'docker ps -q --filter name="tensorflow_serving" --filter status="running" | xargs -r docker stop',
    # Change to the directory where the models are stored
    'cd "' + os.getenv('DEPLOY_SERVER_PATH') + '"',
    # Remove the existing version of this model if present
    'rm -rf "./prod/' + version + '"',
    # The TensorFlow Serving docker container launched by tools/install_server.sh will read models from the prod directory
    'cp -r "' + remote_staging_path + '" ./prod',
    # Restart the docker container
    'docker restart tensorflow_serving'
)

# Bash commands are joined into a single line with && to ensure they are run in order and quit on first failure
command = ' && '.join(commands)
print('Deployment command commencing: ' + command)

# Initialize the SSH client - in production, you should validate the host fingerprint and use key based authentication
client = paramiko.SSHClient() 
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=os.getenv('DEPLOY_SERVER_HOSTNAME'), username=os.getenv(
    'DEPLOY_SERVER_USERNAME'), password=os.getenv('DEPLOY_SERVER_PASSWORD'))

# Execute the command - if there is an error, raise an exception so the CI/CD pipeline step fails
stdin, stdout, stderr = client.exec_command(command)
print(stdout.read().decode())
err = stderr.read().decode()
stdin.close()  # https://github.com/paramiko/paramiko/issues/1617
if err:
    raise Exception(err)
