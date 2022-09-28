#!/usr/bin/env python
import argparse

import cv2
import numpy as np
import pdsc
import requests
import yaml

from marsvision.path_definitions import CONFIG_PATH
from marsvision.path_definitions import PDSC_TABLE_PATH
from marsvision.pipeline import Model
from marsvision.pipeline import SlidingWindow


def main(model_file, model_mode, num_samples, output_file="marsvision.db"):
    """
        This a runnable script that demonstrates the use of the sliding window pipeline
        with the PDSC API.

        A sample command would look like:

        python run_pipeline.py -n 10 -f model.pt -m "pytorch"

        Where 10 is the number of images desired, model.pt is a pytorch model file,
        and -m is the model type (either "pytorch" or "sklearn").
    """
    # Open config file
    with open(CONFIG_PATH) as yaml_cfg:
        config = yaml.safe_load(yaml_cfg)
        config_window = config["random_pipeline_parameters"]

    window_strides = config_window["window_strides"]
    window_sizes = config_window["window_sizes"]

    # Retrieve a random subset of HiRise images.
    # Run the sliding window pipeline on them.
    client = pdsc.PdsClient(PDSC_TABLE_PATH)

    # Query n random metadata grayscale samples.
    random_metadata_list = get_random_metadata_items(int(num_samples), client)

    # Query image data from metadata items.
    image_list = get_images_from_metadata(random_metadata_list)

    # Initialize the model
    model = Model(model_file, model_mode)

    # Run the sliding window pipeline on all of the window sizes specified in the pipeline.
    # In the config file, we specify a list of sizes, 
    # where each window is square.
    for i in range(len(window_sizes)):
        window_size = window_sizes[i]
        window_stride = window_strides[i]
        sliding_window = SlidingWindow(model, output_file, window_size, window_size, window_stride, window_stride)
        sliding_window.sliding_window_predict(image_list, random_metadata_list)
        print("Iteration {0}/{1} complete | Window size: {2}x{2} | Stride x: {3} | Stride y: {3}".format(i + 1,
                                                                                                         len(window_sizes),
                                                                                                         window_size,
                                                                                                         window_stride))


def parse_file_name_to_url(file_name_specification):
    """
        Parses a PDSC file name to a URL.
    """

    # Get path to map projected image by using .NOMAP.
    url_suffix = file_name_specification.split(".")[0] + ".NOMAP.browse.jpg"
    url = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/" + url_suffix
    return url


def get_images_from_metadata(metadata_list):
    """
        Put together image data from a list of PDSC metadata objects.
        Query the HiRISE website for these images.
    """

    image_list = []
    for metadata in metadata_list:
        url = parse_file_name_to_url(metadata.file_name_specification)
        response = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_list.append(image)
    return image_list


# Helper methods to return a random sample of RED metadata items.
def get_random_indices(n, index_range):
    """
        Get n random indices, in range [0, index_range)
        Get a sample of random indices between 0 and n_samples.
    """
    random_indices = (np.random.rand(n) * index_range) // 1
    return random_indices.astype(int)


def get_red_metadata_items(metadata_list):
    """
        Filters out non-RED items.
    
        Returns metadata for only RED items after a query.
    """
    return [m for m in metadata_list if 'RED' in m.product_id]


def get_random_metadata_items(n_samples, client):
    """
        Get a random sample of metadata items
    """
    # Get the full list of metadat, filter out the non-RED items,
    # then return n random samples.
    metadata_list = client.query('hirise_rdr')
    red_metadata_list = get_red_metadata_items(metadata_list)
    random_indices = get_random_indices(n_samples, len(red_metadata_list))
    observation_id_list = np.array([m.observation_id for m in red_metadata_list])
    random_observation_ids = observation_id_list[random_indices]
    random_metadata_items = client.query_by_observation_id("hirise_rdr", random_observation_ids)
    return get_red_metadata_items(random_metadata_items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--num_samples")
    parser.add_argument("-f", "--model_file")
    parser.add_argument("-m", "--model_mode")
    args = parser.parse_args()
    main(**vars(args))
