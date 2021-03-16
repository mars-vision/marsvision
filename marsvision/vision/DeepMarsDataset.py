import os
from PIL import Image
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
  
        # All pre-trained models expect input images normalized in this way:
        #  https://pytorch.org/vision/stable/models.html
        
        # Get image labels
        self.labels = {}
        with open(os.path.join(root_dir, "labels-map-proj.txt")) as f:
            for line in f:
                items = line.split()
                key, value = items[0], items[1]
                self.labels[key] = int(value)


        # Normalize images into [0, 1] with the expected mean and stds.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), # normalize to [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
                
        # Get image filenames
        self.image_dir = os.path.join(root_dir, "map-proj")
        image_names = os.listdir(os.path.join(self.image_dir))
        # Take set difference 
        # to ensure that only labelled images are included
        self.image_names = list(set(image_names) & set(self.labels))
                                  
    def __getitem__(self, idx: int):
        """
            Returns an item in the dataset as a dictionary:
                {'image': image, 'label': label, 'filename': filename}
        """

        # Get a sample as: {'image': image, 'label': label}
        # Expected dimensions:(3, H, W)
        img_name = self.image_names[idx]

        # Use convert because some of the images in the dataset are in grayscale.
        img = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        
        # Apply image preprocessing
        img = self.transform(img)
    
        return {
            "image": img,
            "label": self.labels[self.image_names[idx]],
            "filename": self.image_names[idx]
        }
        
    def __len__(self):
        return len(self.image_names)