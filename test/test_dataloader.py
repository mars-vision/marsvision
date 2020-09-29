from unittest import TestCase
from marsvision.utilities import DataLoader
import os
import numpy as np
import cv2

class TestDataLoader(TestCase):
    def setUp(self):
        # Instantiate loader and set working directory
        current_dir = os.path.dirname(__file__)
        self.test_image_path = os.path.join(current_dir, "test_images_loader")
        self.loader = DataLoader(self.test_image_path, self.test_image_path, "class")
        
    def test_data_reader(self):
        # Manually load a list of images and test against the loader,
        # which will load all images from the test_images_loader folder
        expected_loaded_images = [
           cv2.imread(os.path.join(self.test_image_path, "ESP_011492_1260_RED.NOMAP.browse-Block-2.jpg")),
           cv2.imread(os.path.join(self.test_image_path, "ESP_026455_2460_RED.NOMAP.browse-Block-1.jpg")),
           cv2.imread(os.path.join(self.test_image_path, "ESP_056891_0940_RED.NOMAP.browse-Block-9.jpg")),
        ] 
        self.loader.data_reader()
        self.assertTrue(np.array_equal(expected_loaded_images, self.loader.images))
        
        
