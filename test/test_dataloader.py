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
        self.current_dir = os.path.dirname(__file__)
        self.test_image_path = os.path.join(self.current_dir, "test_data")
        
        # For testing data loader with provided paths
        self.loader = DataLoader(self.test_image_path, self.test_image_path)
        
        # Manually load a list of images and to test against the loader
        self.expected_loaded_images = []
        walk = os.walk(self.test_image_path, topdown=True)
        for root, dirs, files in walk:
            for file in files:
                if file.endswith(".jpg"):
                    img =  cv2.imread(os.path.join(root, file))
                    if img is not None:
                        self.expected_loaded_images.append(img)

        # Calculate expected features which should match with the loader
        self.expected_features = [FeatureExtractor.extract_features(image) for image in self.expected_loaded_images]
        self.loader.data_reader()
        self.loader.data_transformer()
        

    def test_data_reader(self):
        self.assertTrue(np.array_equal(self.expected_loaded_images, self.loader.images))

    def test_data_transformer(self):
        self.assertTrue(np.array_equal(self.expected_features, self.loader.feature_list))

    def test_data_writer(self):
        # Write a test csv file, test against an expected file to ensure a match
        self.loader.run()
        expected_csv_path = os.path.join(self.test_image_path, "output_test.csv")
        expected_df = pd.read_csv(expected_csv_path)
        output_csv_path = os.path.join(self.test_image_path, "output.csv")
        test_df = pd.read_csv(output_csv_path)
        os.remove(output_csv_path)
        np.testing.assert_array_equal(expected_df.sort_values(by="0").values, test_df.sort_values(by="0").values)

    

