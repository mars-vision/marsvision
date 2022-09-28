import os
from unittest import TestCase

import torch

from marsvision.vision import DeepMarsDataset


class test_deep_mars_dataset(TestCase):

    def test_deepmars_dataset(self):
        # Test creating a pytorch dataset with the dataloader
        deep_mars_test_path = os.path.join(os.path.dirname(__file__), "deep_mars_test_data")
        dataset = DeepMarsDataset(deep_mars_test_path)

        # Ensure all of the images were loaded,
        self.assertEqual(len(dataset), len(os.listdir(os.path.join(deep_mars_test_path, "map-proj"))))

        # And that we have tensors
        self.assertTrue(torch.is_tensor(dataset[0]["image"]))
