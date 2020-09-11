# Test cases for feature extractor
from unittest import TestCase
from marsvision.pipeline import FeatureExtractor
import cv2
import os

class TestFeatureExtractor(TestCase):
    # underscores for python
    def test_single_image(self):
        filepath = os.path.join("img", "marsface")
        detector  = cv2.KAZE_create()
        extractor = FeatureExtractor(detector)
        img = cv2.imread(filepath)
        features = extractor.extractFeatures(img)
        print(features)
        self.assertTrue(features)
