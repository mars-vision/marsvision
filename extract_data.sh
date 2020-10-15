#!/bin/bash
conda activate marsvision
cd mars-dataset
python -m marsvision.utilities.DataLoader --i="dust" --c="dust" --f=True
python -m marsvision.utilities.DataLoader --i="featureless" --c="no_dust" --f=True
python -m marsvision.utilities.DataLoader --i="good" --c="no_dust" --f=True
python -m marsvision.utilities.DataLoader --i="miscal" --c="no_dust" --f=True