#!/usr/bin/env bash

# First run:

sudo ./install_dependencies

# then set up a python virtual environment by running:

virtualenv soundClassifier

# after that activate it by running:

source activate

# finally install all the python dependencies by running:

pip3 install -r requirements.txt

