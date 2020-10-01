import numpy as np
import os 
import cv2
import sys
import pandas as pd
import argparse
from marsvision.pipeline import FeatureExtractor as fe

class DataLoader:
    def __init__(self, 
            in_path: str = None,
            out_path: str = None, 
            class_name: str = None): 
        """
            This class is responsible for loading images from an input directory,
            extracting features from them,
            and outputting the processed data as a .csv file. 
            
            A class parameter can be used to classify all images within the inoput folder.

            The csv file will be appended if one already exists, so this script can be invoked on 
            multiple folders to produce an output file with many classes.

            It can be invoked directly via its main function
            with command line arguments.

            Parameters
            ----------
            in_path (str): Optional. The input directory which contains images to be read. Reads from the current working directory if left empty.
            out_path (str): Optional. The output directory to which the csv will be written. Writes to current working directory if left empty.
            class_name (str): Optional. A class name for the input. 

        """
        self.class_name = class_name

        if in_path == None:
            self.in_path = os.getcwd()
        else: 
            self.in_path = in_path

        if out_path == None:
            self.out_path = os.getcwd()
        else: 
            self.out_path = out_path
    def data_reader(self):
        # Use the walk function to step through each folder,
        # and save all jpg files into a list.
        images = []
        walk = os.walk(self.in_path, topdown=True)
        for(root, dirs, files) in walk:
            for file in files:
                if file.endswith(".jpg"):
                    images.append(cv2.imread(os.path.join(root, file)))
        self.images = images

    def data_transformer(self):
        # Use the feature extractor to produce 
        # a list of feature vectors.
        self.feature_vectors = [fe.extract_features(image) for image in self.images]
        

    def data_writer(self):
        # Write features to CSV with path names.
        # Use the feature extractor to retrieve features from images.
        df = pd.DataFrame(data = self.feature_vectors)
        if self.class_name is not None:
            df["class"] = self.class_name
        
        out_file = os.path.join(self.out_path, "output.csv")

        df.to_csv(out_file, mode="a")

    def run(self):
        self.data_reader()
        self.data_transformer()
        self.data_writer()


# Read/write image data with features
# If we run this module directly.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input strings")
    parser.add_argument("--i", help="Input directory")
    parser.add_argument("--o", help="Output directory")
    parser.add_argument("--c", help="Class for input files")
    args = parser.parse_args()
    loader = DataLoader(args.i, args.o, args.c)
    loader.run()





