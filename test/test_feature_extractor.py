from unittest import TestCase
from marsvision.pipeline import FeatureExtractor
import cv2
import os
import numpy as np

class TestFeatureExtractor(TestCase):
    def setUp(self):
        # Expected features for the full marsface.jpg test image
        self.feature_dir = os.path.join(os.path.dirname(__file__), "test_files")
        expected_feature_path = os.path.join(self.feature_dir, "expected_feature_extractor_features.npy")
        self.expected_features = np.load(expected_feature_path)

        # Load up the matrix of expected features from keypoints
        # for our test image from our marsFeatures.npy file
        image_path = os.path.join(os.path.dirname(__file__), "test_files", "marsface.jpg")
        self.img = cv2.imread(image_path)


    def test_feature_extractor(self):
        test_feature_vector = FeatureExtractor.extract_features(self.img)
        self.assertTrue(np.array_equal(self.expected_features, test_feature_vector))