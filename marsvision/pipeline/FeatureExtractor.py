import numpy as np
import cv2
# Constructor:
# Keypoint detector
class FeatureExtractor:
    def __init__(self, detector):
        self.alg = detector

    def getKeypointPoints(self, img):
        keypoints = self.alg.detect(img)
        points = [(round(keypoint.pt[0]), round(keypoint.pt[1])) for keypoint in keypoints]
        return points

    def selectROI(self, img, point, r):
        rowStart = max(0, point[0] - r)
        rowEnd = min(img.shape[0] - 1, point[0] + r)
        colStart = max(0, point[1] - r)
        colEnd = min(img.shape[1] - 1, point[1] + r)
        return img[rowStart : rowEnd, colStart : colEnd]

    def extractFeatures(self, img):
        # Apply filters to each ROI
        # Reduce to means and variances (our features)
        # return a vector of features for each one.
        points = self.getKeypointPoints(img)
        r = 20

        # todo: try averaging values for each keypoint instead
        # in a single vector
        featureMatrix = np.empty((len(points), 4))
        for i in range(len(points)):
            roi = self.selectROI(img, points[i], r)
            canny = cv2.Canny(roi, 50, 100)
            lapl = cv2.Laplacian(roi, cv2.CV_64F)
            vector = [
                np.mean(canny),
                np.var(canny),
                np.mean(lapl),
                np.var(lapl)
            ]
            featureMatrix[i] = vector
        return featureMatrix
            
        
