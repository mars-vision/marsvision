from unittest import TestCase
from marsvision.pipeline import FeatureExtractor
import cv2
import os
import numpy as np

class TestFeatureExtractor(TestCase):
    def setUp(self):
        # Load up the array of expected features 
        # for our test image from our marsFeatures.npy file
        dirname = os.path.dirname(__file__)
        featurePath = os.path.join(dirname, "marsFeatures.npy")
        imagePath = os.path.join(dirname, "marsface.jpg")
        self.img = cv2.imread(imagePath)
        self.marsFeaturesKAZE = np.load(featurePath)
        self.featureMeans = np.mean(self.marsFeaturesKAZE, axis=0)

    def test_single_image_matrix_KAZE(self):
        # Extract features, compare to expected features
        detector  = cv2.KAZE_create()
        extractor = FeatureExtractor(detector)
        testFeatures = extractor.extract_features(self.img)
        self.assertTrue(np.array_equal(self.marsFeaturesKAZE, testFeatures))
    
    def test_single_image_means_KAZE(self):
        # Extract feature means, compare to expected feature means
        detector  = cv2.KAZE_create()
        extractor = FeatureExtractor(detector)
        testMeans = extractor.extract_means(self.img)
        self.assertTrue(np.array_equal(self.featureMeans, testMeans))

