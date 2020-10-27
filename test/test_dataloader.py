from unittest import TestCase
from marsvision.utilities import DataLoader
from marsvision.pipeline import FeatureExtractor
import os
import numpy as np
import cv2
import pandas as pd
from pandas._testing import assert_frame_equal

class TestDataLoader(TestCase):
    def setUp(self):
        # Instantiate loader and set working directory
        current_dir = os.path.dirname(__file__)
        self.test_image_path = os.path.join(current_dir, "test_images_loader")
        self.loader = DataLoader(self.test_image_path, self.test_image_path, "class", True)

        # Manually load a list of images and to test against the loader
        self.expected_loaded_images = [
           cv2.imread(os.path.join(self.test_image_path, "ESP_011492_1260_RED.NOMAP.browse-Block-2.jpg")),
           cv2.imread(os.path.join(self.test_image_path, "ESP_026455_2460_RED.NOMAP.browse-Block-1.jpg")),
           cv2.imread(os.path.join(self.test_image_path, "ESP_056891_0940_RED.NOMAP.browse-Block-9.jpg")),
        ] 

        # Calculate expected features which should match with the loader
        detector  = cv2.ORB_create()
        self.expected_features = [FeatureExtractor.extract_features(image) for image in self.expected_loaded_images]
        self.loader.data_reader()
        self.loader.data_transformer()

    def test_data_reader(self):
        self.assertTrue(np.array_equal(self.expected_loaded_images, self.loader.images))

    def test_data_transformer(self):
        self.assertTrue(np.array_equal(self.expected_features, self.expected_features))

    def test_data_writer(self):
            # Write a test csv file, test against an expected file to ensure a match
            self.loader.data_writer()
            expected_csv_path = os.path.join(self.test_image_path, "output_test.csv")
            expected_df = pd.read_csv(expected_csv_path)
            output_csv_path = os.path.join(self.test_image_path, "output.csv")
            test_df = pd.read_csv(output_csv_path)
            os.remove(output_csv_path)
            assert_frame_equal(expected_df, test_df)
            



    

