from unittest import TestCase
from marsvision.pipeline import KeypointFeatureExtractor
import cv2
import os
import numpy as np

class TestKeypointFeatureExtractor(TestCase):
    def setUp(self):
        # Load up the matrix of expected features from keypoints
        # for our test image from our marsFeatures.npy file
        keypoint_feature_path = os.path.join(os.path.dirname(__file__), "test_files", "mars_test_matrix.npy")
        image_path = os.path.join(os.path.dirname(__file__), "test_files", "marsface.jpg")
        self.img = cv2.imread(image_path)
        self.radius = 20
        self.expected_feature_matrix_KAZE = np.load(keypoint_feature_path)
        self.expected_keypoint_features = np.mean(self.expected_feature_matrix_KAZE, axis=0)

    def test_keypoint_feature_matrix_KAZE(self):
        # Extract features, compare to expected features
        detector  = cv2.KAZE_create()
        extractor = KeypointFeatureExtractor(detector, self.radius)
        test_features = extractor.extract_keypoint_features(self.img)
        self.assertTrue(np.array_equal(self.expected_feature_matrix_KAZE, test_features))

    def test_keypoint_feature_vector_KAZE(self):
        # Extract feature means, compare to expected feature means
        detector = cv2.KAZE_create()
        extractor = KeypointFeatureExtractor(detector, self.radius)
        test_means = extractor.get_means_from_keypoints(self.img)
        self.assertTrue(np.array_equal(self.expected_keypoint_features, test_means))
