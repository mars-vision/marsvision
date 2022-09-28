#!/bin/bash

# Activate our conda env
eval "$(conda shell.bash hook)"
conda activate marsvision

# Run tests
python -m pytest --cov=marsvision/ -W ignore::DeprecationWarning --ignore pdsc

# Run linting
pylint marsvision