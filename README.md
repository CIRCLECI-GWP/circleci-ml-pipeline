# Automating ML Workflows with CircleCI CI/CD Tools for MLOps

This repository provides an example of how a machine learning (ML) workflow can be split into stages and processed using CircleCI’s CI/CD platform for MLOps purposes. This enables the full automation and monitoring of ML workflows while adding additional capabilities like alerts and deployment to auto-scaling cloud environments for heavy workloads.

Given the complexity and bespoke nature of ML models and workflows, this example repository is demonstrative. It shows you how the ML process can be broken down into separate stages and then integrated into a CI/CD pipeline for granular reporting and management based on triggers such as schedules, data updates, and model updates.

Thus, this example uses a simple TensorFlow/Keras-based ML workflow, as the focus is on the CI/CD automation pipeline. Your workflows’ stages and methodology will most likely differ, but the principles will be the same: break down your ML process, automate the training and retraining of data, and let your CI/CD platform handle any failures and notify the responsible parties as part of MLOps best practices.

## Credits

The code in this repository is adapted from the following TensorFlow tutorial:

[https://github.com/tensorflow/tfx/blob/master/docs/tutorials/serving/rest_simple.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/serving/rest_simple.ipynb)


## Definitions

If you’re unfamiliar, here’s a quick rundown of the ML tools used in this repo:

* [TensorFlow](https://www.tensorflow.org/): An open-source ML platform that runs on Python.
* [Keras](https://keras.io/): A deep-learning neural network that runs on top of TensorFlow and provides pre-built ML models.
* [MNIST](https://en.wikipedia.org/wiki/MNIST_database): The Modified National Institute of Standards and Technology database contains datasets containing images and glyphs for testing image processing systems. This example uses data from this database as a test dataset containing images of clothing for training the ML model.


## Usage notes

All commands in this repository should be invoked from its root directory, for example:

    bash ./tools/install_server.sh
    python3 ./ml/1_build.py

There are many comments in the included Python files and Bash scripts that explain what’s going on. Be sure to read them if you run into trouble!

## Repository contents

This repository contains the instructions and scripts required to configure and run the ML workflows manually and a CircleCI configuration that automates the processes:

* The [`ml`](https://github.com/bgmorton/circleci-ml-pipeline/tree/main/ml) directory contains an example ML workflow split across several Python scripts.
    * These scripts rely on a `.env` file with the deployment server details in the root project directory. An example is provided.
* The [`tools`](https://github.com/bgmorton/circleci-ml-pipeline/tree/main/tools) directory contains Bash scripts for setting up the environment to run the ML workflow, testing the workflows locally, and configuring a TensorFlow Serving server.
* Finally, the [`.circleci`](https://github.com/bgmorton/circleci-ml-pipeline/tree/main/.circleci) directory contains the CircleCI `config.yml` that defines the CircleCI pipelines that will call the ML scripts.

### Quick start

Rather than having a long README file, each script in this repository is commented. Start in the `tools` directory to see how to install your ML environment and server, and then read through the `ml` Python scripts to see what they do.

### ML Python scripts

The scripts in the `ml` directory provide the core functionality. Each script contains a stage in a simple ML workflow:

#### 1. Build

* Building an ML model is a multi-step process that involves collecting, validating, and understanding your data and then building a program that can analyze and create insights from that data.
* In our example, the build phase imports and prepares some demo data, ready to train and test an existing[ Keras Sequential model](https://keras.io/guides/sequential_model/) in the next step. In a real-world scenario, you’d supply your own data.

#### 2. Train

* In this step, carefully prepared, highly accurate data with known outcomes is fed to the model so that it can start learning.
* This uses the training data from the build phase.

#### 3. Test

* As the training data has been pre-analyzed and is well understood, we can tell if the trained model is accurate by comparing its output with the already known outcomes.
* In these example scripts, we do this by comparing the output with the testing data imported in the build phase.

#### 4. Package

* This prepares the trained model for use in a separate environment, saving it in a standard format and making it portable so that it can be deployed for use elsewhere.
* It also uploads it to a package store/staging area via SSH. This is done to keep the tutorial focused on the ML side of things. In production, you could use [CircleCI workspaces](https://circleci.com/docs/workspaces/) to share data or use a commonly accessible storage location such as [AWS S3](https://circleci.com/developer/orbs/orb/circleci/aws-s3) to store and retrieve your ML artifacts.

#### 5. Deploy

* This stage involves deploying your trained and packaged model to your production ML environment.
* In this example, the packaged model is uploaded to a directory that[ TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) loads its models from.

#### 6. Retrain

* Deploying a model doesn’t mean it’s finished. New data will arrive that can be used to retrain it to improve its accuracy.
* In this example, a retraining step can replace the training step in this workflow to retrain an existing model rather than creating a new one.
* To see this scheduled workflow in action, you will need to create a branch in your Git repository named `retrain`.
* _Note that in this script, the testing step is designed to fail! This is so that you can see what a failed job looks like when this script is added to a job in CircleCI. Comment out the exception in the Python script to see it succeed._

#### 7. Test deployed model

* Ensuring a successful deployment is important, so this example makes a quick REST call to TensorFlow Serving to ensure that it receives a response.
* The request library will throw an error if the request fails.

## Prerequisites

* A CircleCI account and a GitHub account.
    * See the [CircleCI quickstart guide](https://circleci.com/docs/getting-started/) to learn how to get up and running with both.
    * You can fork the example repository for this tutorial from your own GitHub account and use it as the basis for your CircleCI project.
* A CircleCI self-hosted runner.
    * This can be a local machine or set up as part of an [auto-scaling deployment](https://circleci.com/blog/autoscale-self-hosted-runners-aws/) for larger workloads.
    * See **Python** below for installing additional requirements for the runner.
* Alternatively, you can run this pipeline from CircleCI managed compute cloud resources.
    * To keep this example simple, it’s assumed that it’s running on a self-hosted runner with access to the required network assets (model storage location, TensorFlow Serving server).
    * However, it may be preferable to run this on CircleCI’s infrastructure to take advantage of the available pre-built images and machine classes (see the **GPU** section below) or simply to reduce the infrastructure you have to maintain by using CircleCI’s managed cloud resources.
        * If you are doing this, you will need to make sure CircleCI can access the required network resources by securely exposing them or adding SSH tunnels or [VPN configuration](https://support.circleci.com/hc/en-us/articles/360049397051-How-To-Set-Up-a-VPN-Connection-During-Builds) to your CircleCI pipeline steps or scripts.
* A server with SSH access and Docker installed.
    * See the **TensorFlow Serving** section below for a script to set this up.
    * Your runner should be able to reach this machine on the network.


### Python

The machine that will run these tasks (whether as a CircleCI self-hosted runner or running the scripts locally) must have Python 3 installed. On Ubuntu, run:

    sudo apt install python3 python3-pip python3-venv

This project uses Python [virtual environments](https://docs.python.org/3/library/venv.html), to keep all code and dependencies in the project directory, rather than installing them globally.

### TensorFlow Serving

An additional script located at `tools/install_server.sh` is supplied for spinning up a Docker container running TensorFlow Serving for testing.

Note that you must first install Docker according to its [installation instructions](https://docs.docker.com/engine/install/) for your platform.

You will need to supply the details of the machine this server is running on in your `.env` file so that the Python scripts can access it.

### Test this project locally

To test this project without importing it into CircleCI, you can run `tools/test_build.sh` and `tools/test_retrain.sh` in the Python virtual environment, after creating a `.env` file with the necessary configuration as shown in `.env.example`.

You can install a virtual environment and the required Python packages by running the install script located at `tools/install.sh` (only required for manual testing - the pipeline will call it when required).

    # Use the source command to execute install.sh so that the virtual environment is activated for the current session
    source ./tools/install.sh

    # Run the pipeline scripts for local testing
    bash ./tools/test_build.sh
    bash ./tools/test_retrain.sh

    # Deactivate the virtual environment
    deactivate

The Python packages required by the example ML scripts are:

    tensorflow 
    numpy 
    matplotlib 
    pysftp 
    python-dotenv 
    paramiko 
    requests 

These are installed with their dependencies by `install.sh` into the virtual environment.

## Setting up the project in CircleCI

You will need to fork this repository and [import it into CircleCI](https://circleci.com/docs/create-project/).

### Setting environment variables

You must set the following[ environment variables](https://circleci.com/docs/env-vars/) in CircleCI, as they will be used to generate the `.env` file that the Python scripts read credentials from when the pipeline is executed:

* `DEPLOY_SERVER_HOSTNAME`
* `DEPLOY_SERVER_USERNAME`
* `DEPLOY_SERVER_PASSWORD`
* `DEPLOY_SERVER_PATH`

### Self-hosted runner details

You will need to update the included CircleCI configuration to replace `RUNNER_NAMESPACE/RUNNER_RESOURCE_CLASS` with the details of your own runners. The [CircleCI documentation](https://circleci.com/docs/runner-overview/) explains how to install a self-hosted running on many platforms.

### Using CircleCI

The included CircleCI configuration in `.circleci/config.yml` will run the included scripts as a CI/CD pipeline. You can build on this example to experiment with different [CircleCI features](https://circleci.com/docs/).

If a job fails, you can rapidly respond and confirm the issue in the CircleCI UI by [rerunning only the failed parts of your workflow](https://circleci.com/docs/workflows/#rerunning-a-workflows-failed-jobs).[ ](https://circleci.com/docs/workflows/#rerunning-a-workflows-failed-jobs)

CircleCI requires a valid configuration to run. You can use the CircleCI web interface to edit your `.circleci/config.yml` file, which will include linting and show you any schema problems. Alternatively, you can use the [CircleCI command line tools](https://circleci.com/docs/local-cli/) to [validate your configuration](https://circleci.com/docs/how-to-use-the-circleci-local-cli/#validate-a-circleci-config) locally.


### Onwards!

This example gives an overview of the CircleCI CI/CD functionality that is beneficial to MLOps and automating ML workflows. Once you have experimented with what CircleCI can do with this example, you can start breaking down and automating your own ML workflows. You can also build CircleCI configurations for them that implement the functionality displayed here, such as[ scheduling runs](https://circleci.com/docs/scheduled-pipelines/),[ conditional logic](https://support.circleci.com/hc/en-us/articles/360043638052-Conditional-steps-in-jobs-and-conditional-workflows),[ deploying after approval](https://circleci.com/blog/deploying-with-approvals/), and[ triggering notifications](https://circleci.com/docs/notifications/) based on the results of your pipelines.
