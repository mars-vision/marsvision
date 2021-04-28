#!/bin/bash
conda create --name marsvision
source activate marsvision
conda install -y --quiet --yes numpy pandas pytorch pytest pytest-cov
conda install -y --quiet --yes pillow tensorboard scikit-learn
conda install -y --quiet --yes mypy scipy matplotlib
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda clean -tipsy
pip install pytorch-lightning
pip install coveralls
pip install sphinx
pip install pytorch_sphinx_theme
pip install opencv-python
pip install mypy
pip install exif
cd pdsc
pip install .
mkdir pdsc_tables
cd pdsc_tables
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB > /dev/null >2&1
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL > /dev/null >2&1
cd ..
pdsc_ingest ./pdsc_tables/RDRCUMINDEX.LBL ./pdsc_tables