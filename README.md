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

Clone the directory and run this command to set up the PDSC tables:

<code>./install_pdsc.sh</code>

----
## Running the sliding window pipeline

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
