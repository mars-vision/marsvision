marsvision
======

`marsvision` is an open-source library for automatically detecting surface features on Mars.

[![Build Status](https://travis-ci.org/mars-vision/marsvision.svg?branch=master)](https://travis-ci.org/mars-vision/marsvision)
[![Coverage Status](https://coveralls.io/repos/github/mars-vision/marsvision/badge.svg?branch=master)](https://coveralls.io/github/mars-vision/marsvision?branch=master)

----- 
## Overview
MarsVision is an open source computer vision package containing tools for training and  evaluating Scikit-Learn models and PyTorch CNN models. We implemented a sliding window pipeline, where we can run a sliding window across images for classification of images from PDS.

This package makes use of the PDSC: Planetary Data System Coincidences package to retrieve metadata (e.g. latitudes, longitudes) for classified windows.

[See PDSC here](https://github.com/JPLMLIA/pdsc)


Definitions for Pytorch datasets and models are in the vision directory. Notebooks from our model training and evaluations are in the prototyping directory.

-----
## Installation

Clone the directory and run this command to set up a conda environment and the PDSC package:

<code>./install_env.sh</code>

----
## Training a CNN
This package was written to work with the Deep Mars dataset available (here), though the pytorch code in the model class can be extended to work with any dataset.

The code to train a model looks like this:

<code>
from marsvision.pipeline import Model

from marsvision.vision import ModelDefinitions
alex_model = ModelDefinitions.alexnet_grayscale()
model = Model(alex_model, "pytorch", 
              dataset_root_directory=r"/content/hirise-map-proj"
             )
test_results = model.train_and_test_pytorchcnn()
</code>

Where the root directory is the path to the DeepMars dataset. The definitions for the Dataset class and the model class are the vision folder.

A dictionary containing evaluation results indexed by epoch (including loss, accuracy, precisions, and recalls) will be returned by the training method.

Hyparameters such as number of epochs and learning rate are in the config file.

To write the trained CNN and evaluation results to a file, run the following code:

<code>
model.save_model("MarsVisionSampleCNN.pt")
model.save_results("MarsVisionSampleResults.p")
</code>

----
## Running the sliding window pipeline with a trained model

We wrote a script that runs the sliding window pipeline using a model and writes the results to a database.

The output directory of the cropped windows is specified in the configuration file: marsvision/config.yml.


To run the sample pipeline, run the following script:

<code>bin/run_pipeline_random -n 10 -f MarsVisionSampleCNN.pt -m "pytorch"</code>

The arguments are as follows:

-n: The number of random sample images to process.

-f: Path to the model file. Can either be a pickled SKLearn file (.p) or a Pytorch file (.pt).

-m: Set this to either "pytorch" or "sklearn" depending on the type of model being passed to the script.

---


Documentation
-------------

The documentation for ``marsvision`` is available [here](https://mars-vision.github.io/marsvision/build/index.html).
