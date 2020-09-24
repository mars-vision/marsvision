import numpy as np
import os 
import cv2
import sys
import pandas as pd
from marsvision.pipeline import FeatureExtractor

class DataLoader:
    def __init__(self, inPath, outPath, className = None):
        self.className = className
        print(outPath)
        self.outPath = os.path.join(outPath, "output_features.csv")
        self.inPath = inPath

    def run(self):
        self.data_reader()
        self.data_writer()

    def data_reader(self):
        # Use the walk function to step through each folder,
        # and save all jpg files into a list.
        images = []
        walk = os.walk(self.inPath, topdown=True)
        for(root, dirs, files) in walk:
            for file in files:
                if file.endswith(".jpg"):
                    images.append(cv2.imread(os.path.join(root, file)))
        self.images = images

    def data_writer(self):
        # Write features to CSV with path names.
        # Use the feature extractor to retrieve features from images.
        detector  = cv2.ORB_create()
        extractor = FeatureExtractor(detector)
        featureList = [extractor.extract_means(image) for image in self.images]
        df = pd.DataFrame(data = featureList)
        df["class"] = className
        df.to_csv(self.outPath, mode="a")


# Read/write image data with features
# If we run this script directly.
# TODO: Pick up command line arguments in a better way
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("No input folder specified.")
        sys.exit()

    inputDir = os.path.join(os.getcwd(), sys.argv[1])
    
    if len(sys.argv) <= 2:
        print("No output folder specified. Outputting to current directory.")
        outputDir = os.getcwd()
    else:
        outputDir = os.path.join(os.getcwd(), sys.argv[2])

    if len(sys.argv) <= 3:
        className = None
    else: 
        className = sys.argv[3]

    loader = DataLoader(inputDir, outputDir, className)
    loader.run()





