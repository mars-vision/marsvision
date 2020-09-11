from FeatureExtractor import FeatureExtractor
import cv2
import os


filepath = os.path.join("img", "marsface")
detector  = cv2.KAZE_create()
extractor = FeatureExtractor(detector)
img = cv2.imread(filepath)
features = extractor.extractFeatures(img)
print(features)
