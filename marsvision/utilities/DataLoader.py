import numpy as np
import os 
import cv2
import sys
import pandas as pd
import argparse
from marsvision.pipeline import FeatureExtractor
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    # This might be nicer with keyword arguments
    def __init__(self, 
            in_path: str = None,
            out_path: str = None): 
        """
            This class is responsible for loading images from an input directory,
            extracting features from them, and outputting the processed data as a .csv file. 
            
            A class parameter can be used to classify all images within the input folder.

            It can be invoked directly via the command line.

            ---
            
            Parameters:

            in_path (str): Optional. The input directory which contains images to be read. Reads from the current working directory if left empty.

            out_path (str): Optional. The output directory to which the csv will be written. Writes to current working directory if left empty.

            detector_name(string): Optional. Name of the detector to use to detect keypoints.

            ---

            Command Line Arguments:

            --i: Input directory. Default: current working directory

            --o: Output directory. Default: output to current working directory.

            --c: Class for input files. Default: use containing folder as class name.

            --f: Boolean, whether to include the file name or not. Default: True

        """
        # Set values based on whether default parameters are set
        if in_path == None:
            self.in_path = os.getcwd()
        else: 
            self.in_path = in_path

        if out_path == None:
            self.out_path = os.getcwd()
        else: 
            self.out_path = out_path

    def data_reader(self):
        """
            Walk through a folder and load images, file names, and folder names into memory
            as member variables.
            
            All .jpg images in the working directory,
            and all subdirectories are loaded.

            This function updates the self.images, self.file_names, and self.labels members with the loaded data.

        """

        # Use the walk function to step through each folder,
        # and save all jpg files into a list.
        images = []
        file_names = []
        labels = []
        walk = os.walk(self.in_path, topdown=True)
        for root, dirs, files in walk:
            for file in files:
                if file.endswith(".jpg"):
                    img =  cv2.imread(os.path.join(root, file))
                    if img is not None:
                        images.append(img)
                        file_names.append(file)
                        labels.append(os.path.basename(root))

        self.images = images
        self.file_names = file_names
        self.labels = labels


    def data_transformer(self):
        """
            Use the FeatureExtractor module to load
            a vector of features into memory as a member variable.

            Creates a Pandas dataframe from the extracted features, and write the data to a .csv file ("output.csv"),
            to the path which was specified in the constructor.

            Set columns depending on user preferences:
            If a class is defined, write to the class to a class column.
            If no class is defines, the containing folder name will be used as the class in the class column.
            If file names are desired, file names are written to a file_name column.

        """
        # Use the feature extractor to produce 
        # a list of feature vectors.

        self.feature_list = [FeatureExtractor.extract_features(image) for image in self.images]
        self.df = pd.DataFrame(data = self.feature_list)
        self.df["class"] = self.labels
        self.df["file_name"] = self.file_names
        LE = LabelEncoder()
        self.df["class_code"] = LE.fit_transform(self.df["class"])


    def data_writer(self):
        """
            Write the constructed dataframe to an output file ("output.csv")

        """
        # Write features to CSV with path names.
        # Use the feature extractor to retrieve features from images.
        out_file = os.path.join(self.out_path, "output.csv")
        self.df.to_csv(out_file, index=False)

    def run(self):
        """
            When called, run() will execute the data_reader(), data_transformer(), and data_writer() 
            functions in order.

            The result will be an output .csv file either in the directory from which the script is run,
            or in the user specified directory.
        """
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
    parser.add_argument("--f", default=True, nargs="?", help="(Boolean) Whether to include the file name or not.")
    args = parser.parse_args()
    loader = DataLoader(args.i, args.o, args.c, args.f)
    loader.run()





