import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from torch import Tensor

class DeepMarsDataset(Dataset):
    def __init__(self, root_dir: str):
        """
           Initialize transforms. Extract image labels from the dataset in root_dir. 

           ----

           Parameters:

           root_dir (str): Root directory of the Deep Mars dataset.
        """
        
        # All pre-trained models expect input images normalized in this way.
        # Source: https://pytorch.org/vision/stable/models.html

        # TODO: Crop images to 256x256 -- may need to read images with PIL instead in here
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        # Get image labels
        self.labels = {}
        with open(os.path.join(root_dir, "labels-map-proj.txt")) as f:
            for line in f:
                items = line.split()
                key, value = items[0], items[1]
                self.labels[key] = int(value)
                
        # Get image filenames
        self.image_dir = os.path.join(root_dir, "map-proj")
        image_names = os.listdir(os.path.join(self.image_dir))
        # Take set difference 
        # to ensure that only labelled images are included
        self.image_names = list(set(image_names) & set(self.labels))
                                  
    def __getitem__(self, idx: int):
        """
            Returns an item in the dataset as a dictionary:
                {'image': image, 'label': label}
        """

        # Get a sample as: {'image': image, 'label': label}
        # Return an image with the dimensions 3 x W x H
        # Because PyTorch models expect these dimensions as inputs.
        # Transpose dimensions:
        # (W, H, 3) --> (3, W, H)
        img_name = self.image_names[idx]
        img = Tensor(
            cv2.imread(os.path.join(self.image_dir, img_name))
        ).transpose(0, 2)
        
        # Apply normalize
        img = self.normalize(img)
    
        return {
            "image": img,
            "label": self.labels[self.image_names[idx]]
        }
        
    def __len__(self):
        return len(self.image_names)