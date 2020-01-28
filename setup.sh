#!/usr/bin/env bash

# First run:

sudo ./install_dependencies.sh

# then set up a python virtual environment by running:

virtualenv --python=python3 soundClassifier

# after that activate it by running:

source activate

# finally install all the python dependencies by running:

pip3 install -r requirements.txt

