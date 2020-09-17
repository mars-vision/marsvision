import sys
import os
from marsvision import pipeline

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("No input folder specified.")
        sys.exit()

    inputDir = os.path.join(os.getcwd(), sys.argv[1])

    if len(sys.argv) <= 2:
        print("No output folder specified. Outputting to current directory.")
        outputDir = os.getcwd()
    else:
        outputDir = sys.argv[2]

    dataLists = pipeline.dataloader(inputDir)