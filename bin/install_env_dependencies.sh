#!/bin/bash
set -e

# Packages installation
conda env create --name marsvision --file=environment.yml

# PDSC Installation
rm -rf ../data/pdsc_tables
mkdir ../data/pdsc_tables
cd ../data/pdsc_tables
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB -o RDRCUMINDEX.TAB
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL -o RDRCUMINDEX.LBL
cd ..
eval "$(conda shell.bash hook)"
conda activate marsvision
pdsc_ingest ./pdsc_tables/RDRCUMINDEX.LBL ./pdsc_tables