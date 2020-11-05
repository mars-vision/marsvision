from unittest import TestCase
from marsvision.pipeline import FeatureExtractor
import cv2
import os
import numpy as np

class TestFeatureExtractor(TestCase):
    def setUp(self):
        # Expected features for the full marsface.jpg test image
        self.current_dir = os.path.dirname(__file__)
        expected_feature_path = os.path.join(self.current_dir, "expected_feature_extractor_features.npy")
        self.expected_features = np.load(expected_feature_path)

        # Load up the matrix of expected features from keypoints
        # for our test image from our marsFeatures.npy file
        dirname = os.path.dirname(__file__)
        image_path = os.path.join(dirname, "marsface.jpg")
        self.img = cv2.imread(image_path)


    def test_feature_extractor(self):
        test_feature_vector = FeatureExtractor.extract_features(self.img)
        self.assertTrue(np.array_equal(self.expected_features, test_feature_vector))