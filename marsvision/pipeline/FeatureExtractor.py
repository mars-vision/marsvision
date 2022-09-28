import cv2
import numpy as np


class FeatureExtractor:
    num_features = 6

    @staticmethod
    def extract_features(img,
                         canny_threshold_1: int = 50,
                         canny_threshold_2: int = 100):
        """
        A Feature extractor that takes an image as an input
        and outputs a vector of features by reducing the dimensionality
        of the input image to a vector.

        Applies canny and laplacian filters on an input image,
        takes means and variances of those filters,
        and means and variances of the image itself.
        Return the vector of features.

        ---

        Parameters:
        
        img (openCV image): Input image to extractor features
        canny_threshold_1 (int): OpenCV Canny Threshold 1 for canny detector
        canny_threshold_2 (int):  OpenCV Canny Theshold 2 for canny detector
        """
        feature_vector = []
        # Exception thrown when image is null
        try:
            img = np.array(img)
            canny = cv2.Canny(img, canny_threshold_1, canny_threshold_2)
            lapl = cv2.Laplacian(img, cv2.CV_64F)
            feature_vector = [
                np.mean(canny),
                np.var(canny),
                np.mean(lapl),
                np.var(lapl),
                np.mean(img),
                np.var(img)
            ]
        except:
            print("Invalid image : " + str(img))

        return feature_vector
