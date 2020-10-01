from unittest import TestCase
from marsvision.pipeline import FeatureExtractor as fe
import cv2
import os
import numpy as np

class TestFeatureExtractor(TestCase):
    def setUp(self):
        # Load up the matrix of expected features from keypoints
        # for our test image from our marsFeatures.npy file
        dirname = os.path.dirname(__file__)
        image_path = os.path.join(dirname, "marsface.jpg")
        self.img = cv2.imread(image_path)

        # take the means of the columns to reduce the feature matrix to a single vector
        keypoint_feature_path = os.path.join(dirname, "marsFeatures.npy")
        self.expected_feature_matrix_KAZE = np.load(keypoint_feature_path)
        self.expected_keypoint_features = np.mean(self.expected_feature_matrix_KAZE, axis=0)

        # Expected features for the full marsface.jpg test image
        self.expected_features = [
            22.536013333862456, 
            5238.8115031509, 
            -0.0013294178340410333, 
            58.33776957278075, 
            123.72320330171833, 
            3634.7277409035864
        ]

    def test_keypoint_feature_matrix_KAZE(self):
        # Extract features, compare to expected features
        detector  = cv2.KAZE_create()
        extractor = fe(detector)
        test_features = extractor.extract_matrix_keypoints(self.img)
        self.assertTrue(np.array_equal(self.expected_feature_matrix_KAZE, test_features))
    
    def test_keypoint_feature_vector_KAZE(self):
        # Extract feature means, compare to expected feature means
        detector = cv2.KAZE_create()
        extractor = fe(detector)
        test_means = extractor.extract_means_keypoints(self.img)
        print(self.expected_keypoint_features, test_means)
        self.assertTrue(np.array_equal(self.expected_keypoint_features, test_means))

    def test_feature_vector(self):
        test_feature_vector = fe.extract_features(self.img)
        self.assertTrue(np.array_equal(self.expected_features, test_feature_vector))