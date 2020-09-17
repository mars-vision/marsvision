import numpy as np
import os 
import cv2

def dataloader(path):

    # Use the walk function to step through each folder,
    # and save all jpg files into a dictionary.
    # key = folder name, value = list of image paths in folder
    folders = {}
    walk = os.walk(path, topdown=True)
    for(root, dirs, files) in walk:
        folderName = os.path.basename(root)
        folders[folderName] = []
        for file in files:
            if file.endswith(".jpg"):
                folders[folderName].append(os.path.join(root, file))
    print(folders)