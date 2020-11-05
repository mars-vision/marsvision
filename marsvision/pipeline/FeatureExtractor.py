import numpy as np
import cv2
class FeatureExtractor:
    """
    A Feature extractor that takes an image as an input
    and outputs a vector of features by reducing the dimensionality
    of the input image to a vector.
    """

    ## Canny parameters that we can tune if we like
    num_features = 6
    canny_threshold_1 = 50
    canny_threshold_2 = 100

    def extract_features(img):
        """
        Applies canny and laplacian filters on an input image,
        takes means and variances of those filters,
        and means and variances of the image itself.
        Return the vector of features.

        ---

        Parameters:
        img (openCV image): Input image to extractor features
        """
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
