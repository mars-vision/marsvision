import numpy as np
import sklearn
import torch.optim as optim
import torch.save as save
import pickle

class Model:
    Model.PYTORCH = "pytorch"
    Model.SKLEARN = "sklearn"

    def __init__(self, 
        model_type: str = MODEL.PYTORCH,
        model,
        training_data,
        training_labels,
        num_epochs: int = 2)
        self.model_type = model_type
        self.model = model

    def train_model():
        if model_type == MODEL.PYTORCH:
            train_pytorch()

        if model_type == MODEL.SKLEARN
            model.fit()
        

    def train_pytorch():
        # Train a pytorch model given 
        # training data and class labels
        for repeat in range(num_epochs):
            for train_data in in_data:
                output = model(input_data)
                optimizer.zero_grad()
                loss = criterion(output, training_labels)
                loss.backward()
                optimizer.step()

    def save_model(out_filename: str = "output.p"):
        if model_type == MODEL.SKLEARN:
            with open(out_filename) as out_file:
                pickle.dump(self.model, out_file)
                
        if model_type == MODEL.PYTORCH:
            save(model, out_filename)


    def load_model();
        #load model from file

    

    
