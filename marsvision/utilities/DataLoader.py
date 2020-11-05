import numpy as np
import os 
import cv2
import sys
import pandas as pd
import argparse
from marsvision.pipeline import FeatureExtractor

class DataLoader:
    # This might be nicer with keyword arguments
    def __init__(self, 
            in_path: str = None,
            out_path: str = None, 
            class_name: str = None,
            include_filename: bool = True,
            detector_name: str = "ORB"): 
        """
            This class is responsible for loading images from an input directory,
            extracting features from them, and outputting the processed data as a .csv file. 
            
            A class parameter can be used to classify all images within the input folder.

            It can be invoked directly via the command line.

            ---
            
            Parameters:

            in_path (str): Optional. The input directory which contains images to be read. Reads from the current working directory if left empty.

            out_path (str): Optional. The output directory to which the csv will be written. Writes to current working directory if left empty.

            class_name (str): Optional. A class name for the input.

            include_filename(bool): Optional. Whether to include the file name. False by default. 

            detector_name(string): Optional. Name of the detector to use to detect keypoints.

            ---

            Command Line Arguments:

            --i: Input directory. Default: current working directory

            --o: Output directory. Default: output to current working directory.

            --c: Class for input files. Default: use containing folder as class name.

            --f: Boolean, whether to include the file name or not. Default: True

            
            

        """
        self.class_name = class_name

        # Set values based on whether default parameters are set
        if in_path == None:
            self.in_path = os.getcwd()
        else: 
            self.in_path = in_path

        if out_path == None:
            self.out_path = os.getcwd()
        else: 
            self.out_path = out_path

        self.include_filename = include_filename

        self.detector_name = detector_name

    def data_reader(self):
        """
            Walk through a folder and load images, file names, and folder names into memory
            as member variables.
            
            All .jpg images in the working directory,
            and all subdirectories are loaded.

            This function updates the self.images, self.file_names, and self.folder_names members with the loaded data.

        """

        # Use the walk function to step through each folder,
        # and save all jpg files into a list.
        images = []
        file_names = []
        folder_names = []
        walk = os.walk(self.in_path, topdown=True)
        for root, dirs, files in walk:
            for file in files:
                if file.endswith(".jpg"):
                    img =  cv2.imread(os.path.join(root, file))
                    if img is not None:
                        images.append(img)
                        file_names.append(file)
                        folder_names.append(os.path.basename(root))

        self.images = images
        self.file_names = file_names
        self.folder_names = folder_names

    def data_transformer(self):
        """
            Use the FeatureExtractor module to load
            a vector of features into memory as a member variable.
        """
        # Use the feature extractor to produce 
        # a list of feature vectors.
        detector  = cv2.ORB_create()
        self.feature_list = [FeatureExtractor.extract_features(image) for image in self.images]

    def data_writer(self):
        """
            Creates a Pandas dataframe from the extracted features, and write the data to a .csv file ("output.csv"),
            to the path which was specified in the constructor.
            
            Set columns depending on user preferences:
            If a class is defined, write to the class to a class column.
            If no class is defines, the containing folder name will be used as the class in the class column.
            If file names are desired, file names are written to a file_name column.
        """
        # Write features to CSV with path names.
        # Use the feature extractor to retrieve features from images.
        df = pd.DataFrame(data = self.feature_list)
        
        if self.class_name is not None:
            df["class"] = self.class_name
        else:
            df["class"] = self.folder_names

        if self.include_filename:
            df["file_name"] = self.file_names

        out_file = os.path.join(self.out_path, "output.csv")
        df.to_csv(out_file, header=False, index=False)

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





