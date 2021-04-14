from marsvision.pipeline import Model
from marsvision.pipeline import SlidingWindow
import argparse
import requests
import pdsc
import cv2
import numpy as np

def main(model_file, model_mode):
    # Set up PDSC images here.
    client = pdsc.PdsClient("pdsc\pdsc_tables")
    metadata_list = client.query_by_observation_id('hirise_rdr', [
        'PSP_006569_1135',
        'PSP_006570_1820',
        'PSP_006570_1820',
        'PSP_006571_1905',
        'PSP_006571_1905',
        'PSP_006572_1665',
        'PSP_006572_1665',
        'PSP_006573_1560',
        'PSP_006573_1560',
        'PSP_006574_1375',
        'PSP_006574_1375',
        'PSP_006575_1280',
        'PSP_006575_1280'
        ])
    image_list = get_images_from_metadata(metadata_list)
    model = Model(model_file, model_mode)
    sliding_window = SlidingWindow(model, "marsvision.db", 256, 256, 256, 256)
    sliding_window.sliding_window_predict(image_list, metadata_list)

def parse_file_name_to_url(file_name_specification):
    url_suffix = file_name_specification.split(".")[0] + ".abrowse.jpg"
    url = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/" + url_suffix
    return url

def get_images_from_metadata(metadata_list):
    image_list = []
    for metadata in metadata_list: 
        url = parse_file_name_to_url(metadata.file_name_specification)
        response = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_list.append(image)
    return image_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--model_file")
    parser.add_argument("-m", "--model_mode")

    args = parser.parse_args()
    main(**vars(args))
