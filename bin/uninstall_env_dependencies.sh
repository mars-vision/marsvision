#!/bin/bash
set -e

# Remove conda environment
conda env remove -n marsvision

# Remove pdsc data files
rm -rf data/pdsc_tables
