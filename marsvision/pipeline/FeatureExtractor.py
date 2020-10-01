import numpy as np
import cv2


class FeatureExtractor:
    # TODO: Refactor this to use img as a field instead of passing it around functions.
    def __init__(self, detector):
        self.alg = detector

    def get_keypoint_points(self, img):
        keypoints = self.alg.detect(img)
        return [(round(keypoint.pt[0]), round(keypoint.pt[1])) for keypoint in keypoints]

    def select_roi(self, img, point, r):
        row_start = max(0, point[0] - r)
        row_end = min(img.shape[0] - 1, point[0] + r)
        colStart = max(0, point[1] - r)
        colEnd = min(img.shape[1] - 1, point[1] + r)
        return img[row_start : row_end, colStart : colEnd]

    def extract_matrix_keypoints(self, img):
        # Apply filters to each ROI
        # Reduce to means and variances
        # return a vector of features for each one.
        points = self.get_keypoint_points(img)
        radius = 20
        feature_matrix = np.empty((len(points), 4))

        for i in range(len(points)):
            roi = self.select_roi(img, points[i], radius)
            canny = cv2.Canny(roi, 50, 100)
            lapl = cv2.Laplacian(roi, cv2.CV_64F)
            vector = [
                np.mean(canny),
                np.var(canny),
                np.mean(lapl),
                np.var(lapl)
            ]
            feature_matrix[i] = vector

        return feature_matrix

    def extract_means_keypoints(self, img):
        feature_matrix = self.extract_matrix_keypoints(img)
        feature_means = np.mean(feature_matrix, axis=0)
        return feature_means

    @staticmethod
    def extract_features(img):
        # Extract features from images without keypoints.
        # Use the raw image, and extract these features.
        img = np.array(img)
        canny = cv2.Canny(img, 50, 100)
        lapl = cv2.Laplacian(img, cv2.CV_64F)
        feature_vector = [
            np.mean(canny),
            np.var(canny),
            np.mean(lapl),
            np.var(lapl),
            np.mean(img),
            np.var(img)
        ]
        return feature_vector




        
