from unittest import TestCase
from marsvision.vision import DeepMarsDataset
from torch import Tensor
import torch
import os
import cv2

class test_deep_mars_dataset(TestCase):

    def test_deepmars_dataloader(self):
        # Test creating a pytorch dataset with the dataloader
        deep_mars_test_path = os.path.join(os.path.dirname(__file__), "deep_mars_test_data")
        dataset = DeepMarsDataset(deep_mars_test_path)

        
        # Read the first line in the labels file
        with open(os.path.join(deep_mars_test_path, "labels-map-proj.txt")) as f:
            line_items = f.readline().split()
            first_filename, first_label = line_items[0], line_items[1]

        first_label = int(first_label)

        # Pytorch normalize object attached to this dataloader
        normalize = dataset.normalize

        # Apply transform to this tensor
        test_img_path = os.path.join(deep_mars_test_path, "map-proj", first_filename)
        test_img = cv2.imread(test_img_path)
        
        expected_tensor = normalize(
            Tensor(
                test_img
            ).transpose(0, 2)
        )

        test_tensor = dataset[0]["image"]
        self.assertTrue(torch.equal(expected_tensor, test_tensor))

        test_label = dataset[0]["label"]
        self.assertEqual(test_label, first_label)
        