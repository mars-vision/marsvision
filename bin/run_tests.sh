#!/bin/bash

cd ..

# Activate our conda env
eval "$(conda shell.bash hook)"
conda activate marsvision

# Run tests
python -m pytest --cov=marsvision/ -W ignore::DeprecationWarning --ignore pdsc

# Run static code analysis
pylint marsvision
mypy marsvision