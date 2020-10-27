#!/bin/bash
dir_path=$(dirname $(realpath $0))
cd mars-dataset
python $dir_path/marsvision/utilities/DataLoader.py --i=dust --c=dust --f
python $dir_path/marsvision/utilities/DataLoader.py --i=featureless --c=no_dust --f
python $dir_path/marsvision/utilities/DataLoader.py --i=good --c=no_dust --f
python $dir_path/marsvision/utilities/DataLoader.py --i=miscal --c=no_dust --f