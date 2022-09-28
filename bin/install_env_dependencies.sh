#!/bin/bash
set -e

# Packages installation
conda create --name marsvision -f environment.yml

# PDSC Installation
mkdir ../data/pdsc_tables
cd ../data/pdsc_tables
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB -o RDRCUMINDEX.TAB
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL -o RDRCUMINDEX.LBL
cd ..
pdsc_ingest ./pdsc_tables/RDRCUMINDEX.LBL ./pdsc_tables