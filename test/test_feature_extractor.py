# Test cases for feature extractor
from unittest import TestCase
from marsvision.pipeline import FeatureExtractor
import cv2
import os
import numpy as np

class TestFeatureExtractor(TestCase):
    def test_single_image(self):
        # Good way of joining file paths
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, "img", "marsface.jpg")
        detector  = cv2.KAZE_create()
        img = cv2.imread(filepath)
        extractor = FeatureExtractor(detector)
        features = extractor.extractFeatures(img)
        print(features, np.shape(features))
