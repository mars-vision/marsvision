from marsvision.pipeline import KeypointFeatureExtractor
import cv2
import numpy as np

# Creates a testing keypoint matrix file
extractor = KeypointFeatureExtractor(cv2.KAZE_create())
img = cv2.imread("marsface.jpg")
test_matrix = extractor.extract_keypoint_features(img, 20)
np.save("mars_test_matrix", test_matrix)

