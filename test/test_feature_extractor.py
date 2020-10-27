from unittest import TestCase
from marsvision.pipeline import FeatureExtractor
import cv2
import os
import numpy as np

class TestFeatureExtractor(TestCase):
    def setUp(self):
        # Expected features for the full marsface.jpg test image
        self.expected_features = [
            22.536013333862456, 
            5238.8115031509, 
            -0.0013294178340410333, 
            58.33776957278075, 
            123.72320330171833, 
            3634.7277409035864
        ]

        # Load up the matrix of expected features from keypoints
        # for our test image from our marsFeatures.npy file
        dirname = os.path.dirname(__file__)
        image_path = os.path.join(dirname, "marsface.jpg")
        self.img = cv2.imread(image_path)


    def test_feature_extractor(self):
        test_feature_vector = FeatureExtractor.extract_features(self.img)
        self.assertTrue(np.array_equal(self.expected_features, test_feature_vector))