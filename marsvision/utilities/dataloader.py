import numpy as np
import os 
import cv2
import sys

def dataloader(self, path):
    # Use the walk function to step through each folder,
    # and save all jpg files into a list.
    images = []
    walk = os.walk(path, topdown=True)
    for(root, dirs, files) in walk:
        for file in files:
            if file.endswith(".jpg"):
                images.append(cv2.imread(os.path.join(root, file)))
    return images

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("No input folder specified.")
        sys.exit()

    inputDir = os.path.join(os.getcwd(), sys.argv[0])
    
    if len(sys.argv) < 2:
        print("No output folder specified. Outputting to current directory.")
        outputDir = os.getcwd()
    else:
        outputDir = os.path.join(os.getcwd(), sys.argv[1])

    dataloader(inputDir, outputDir)





