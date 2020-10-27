import numpy as np
import cv2
class FeatureExtractor:
    """
        Applies canny and laplacian filters on an input image,
        takes means and variances of those filters,
        and means and variances of the image itself.
        Return the vector of features.
    """

    num_features = 6
    canny_threshold_1 = 50
    canny_threshold_2 = 100

    def extract_features(img):
        img = np.array(img)
        canny = cv2.Canny(img, FeatureExtractor.canny_threshold_1, FeatureExtractor.canny_threshold_2)
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
