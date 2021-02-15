#!/bin/bash
conda create --name marsvision
source activate marsvision
conda install -y --quiet --yes numpy pandas pytorch pytest pytest-cov
conda install -y --quiet --yes pillow tensorboard scikit
conda install -y --quiet --yes torchvision mypy scipy matplotlib
conda clean -tipsy
pip install pytorch-lightning
pip install coveralls
pip install sphinx
pip install pytorch_sphinx_theme
pip install opencv-python
pip install mypy
pip install exif