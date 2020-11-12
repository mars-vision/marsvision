import numpy as np
import sklearn
import os
import torch
import torch.optim as optim
import pickle
from marsvision.pipeline import FeatureExtractor
import torch.nn as nn

class ModelTrainer:
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"

    def __init__(self, 
        training_images,
        training_labels,
        model,
        model_type: str = PYTORCH,
        num_epochs: int = 2):
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.model = model
        self.training_images = training_images
        self.training_labels = training_labels
        """
            training_images: Batch of images to train on 
            training_labels: Class labels for the training images

        """

    def train_model(self):
        if self.model_type == ModelTrainer.PYTORCH:
            self.train_pytorch()

        if self.model_type == ModelTrainer.SKLEARN:
            # Extract features from every image in the batch,
            # then fit the sklearn model to these features.
            extracted_features = [FeatureExtractor.extract_features(image) for image in self.training_images]
            self.model.fit(extracted_features, self.training_labels)
        

    def train_pytorch(self):
        # Train a pytorch model given 
        # training data and class labels
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        trainloader = torch.utils.data.DataLoader(self.training_images, batch_size = 4)

        for repeat in range(self.num_epochs):
            for i, batch in enumerate(trainloader, 0):
                in_data, labels
                output = self.model()
                optimizer.zero_grad()
                loss = criterion(output, self.training_labels)
                loss.backward()
                optimizer.step()


    def save_model(self, out_path: str):
        if self.model_type == ModelTrainer.SKLEARN:
            with open(out_path, "wb") as out_file:
                pickle.dump(self.model, out_file)
                
        if self.model_type == ModelTrainer.PYTORCH:
            torch.save(model, out_filename)


    def load_model(self, input_path):
        with open(input_path, 'rb') as in_file:
            self.model = pickle.load(in_file)
        #load model from file




