import numpy as np
import cv2
from marsvision.pipeline import FeatureExtractor

class KeypointFeatureExtractor:
    def __init__(self, detector,
            radius: int = 20):
        """
            This class uses an OpenCV detector to detect keypoints on image and
            returns reduces the area around the keypoints to a vector of numerical features.
            
            These features are means and variances of the region,
            and means and variances of the region after applying Canny and Laplacian filters.
        """
        self.alg = detector
        self.radius = radius

    def get_keypoint_points(self, img):
        """
            Use an OpenCV algorithm to detect keypoints.
        """
        keypoints = self.alg.detect(img)
        return [(round(keypoint.pt[0]), round(keypoint.pt[1])) for keypoint in keypoints]

    def select_roi(self, img, point: list):
        """
            Select the ROI (Region of Interest)
            in a radius around a given point.
        """
        row_start = max(0, point[0] - self.radius)
        row_end = min(img.shape[0] - 1, point[0] + self.radius)
        colStart = max(0, point[1] - self.radius)
        colEnd = min(img.shape[1] - 1, point[1] + self.radius)
        return img[row_start : row_end, colStart : colEnd]

    def extract_keypoint_features(self, img):
        """
            Build a matrix of features from 
            the feature vectors of each keypoint.
        """
        # Apply filters to each ROI
        # Reduce to means and variances
        # return a vector of features for each one.
        points = self.get_keypoint_points(img)
        feature_matrix = np.empty((len(points), FeatureExtractor.num_features))

        for i in range(len(points)):
            roi = self.select_roi(img, points[i])
            feature_matrix[i] = FeatureExtractor.extract_features(roi)

        return feature_matrix

    def get_means_from_keypoints(self, img):
        """
            Reduce the feature matrix to a vector
            by taking the mean of each feature.
        """
        feature_matrix = self.extract_keypoint_features(img)
        feature_means = np.mean(feature_matrix, axis=0)
        return feature_means





        
