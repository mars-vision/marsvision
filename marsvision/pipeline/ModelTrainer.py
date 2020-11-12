import numpy as np
import sklearn
import os
import torch
import torch.optim as optim
import pickle

class ModelTrainer:
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"

    def __init__(self, 
        training_data,
        training_labels,
        model,
        model_type: str = PYTORCH,
        num_epochs: int = 2):
        self.model_type = model_type
        self.model = model
        self.training_data = training_data
        self.training_labels = training_labels

    def train_model(self):
        if self.model_type == ModelTrainer.PYTORCH:
            train_pytorch()

        if self.model_type == ModelTrainer.SKLEARN:
            self.model.fit(self.training_data, self.training_labels)
        

    def train_pytorch(self):
        # Train a pytorch model given 
        # training data and class labels
        for repeat in range(num_epochs):
            for train_data in in_data:
                output = model(input_data)
                optimizer.zero_grad()
                loss = criterion(output, training_labels)
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




