#!/bin/bash
conda create --name marsvision
source activate computer-vision
conda install -y --quiet --yes numpy pandas pytorch pytest pytest-cov
conda install -y --quiet --yes pillow tensorboard
conda install -y --quiet --yes torchvision mypy scipy
conda clean -tipsy
pip install pytorch-lightning
pip install coveralls
pip install sphinx
pip install pytorch_sphinx_theme
pip install matplotlib
pip install opencv-python
pip install scikit-learn
