import numpy as np
import sklearn
import os
import torch
import pickle
from marsvision.pipeline.FeatureExtractor import *
from sklearn.model_selection import cross_validate, StratifiedKFold
from typing import List
import torch
import torchvision

class Model:
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"

    def __init__(self, 
        model,
        model_type: str = PYTORCH,
        **kwargs):
        """
            Model class that serves as an abstract wrapper for either an sklearn or pytorch model.

            Contains methods for making predictions, and for cross validating the model and writing results to a file.
            --------
            
            Parameters:

            training_images (numpy.ndarray): Batch of images to train on 
            training_labels: Class labels for the training images
            model: Either an sklearn machine learning model, or a pytorch neural network. Can be a path to a file or a model object.
            model_type (str): String identifier for the type of model. Determines how the model will be trained in this class.
        """

        if "training_images" in kwargs:
            self.training_images = kwargs["training_images"]

        if "training_labels" in kwargs:
            self.training_labels = kwargs["training_labels"]

        self.model_type = model_type
        if type(model) == str:
            self.load_model(model, self.model_type)
        else:
            self.model = model
        
        # Initialize extracted features to none; use this member when we use the sklearn model
        self.extracted_features = None

    def predict(self, image_list: np.ndarray):
        """
            Run inference using self.model on a list of images using the currently instantiated model.
            
            This model can either be an sklearn model or a pytorch model.

            Returns a list of inferences.
            
            ---

            Parameters:
                image_list (List[np.ndarray]): Batch of images to run inference on with this model.
        """

        # Handle the case of a single image by casting it as a list with itself in it
        if len(image_list.shape) == 3:
            image_list = [image_list]

        if self.model_type == Model.SKLEARN:
            # Iterate images in batch,
            # Extract features,
            # Use self.model for inference on each
            # Return a list of inferences
            image_feature_list = []
            for image in image_list:
                image_feature_list.append(FeatureExtractor.extract_features(image))
            inference_list = self.model.predict(image_feature_list)
            return list(map(int, inference_list))

        elif self.model_type == Model.PYTORCH: # pragma: no cover
            # TODO: Implement pytorch for model class
            Exception("Invalid model specified in marsvision.pipeline.Model")

    def set_training_data(self, training_images: np.ndarray, training_labels: List[str]):
        """
            Setter for training image data.

            ---

            Parameters: 

            training_images (self): List of images to train the model on. Numpy is expected to be as follows: (image count, height, image width, channels)
            training_labels (self): Labels associated with training images. Should be a list parallel to the list of training images.

        """
        self.training_images = training_images
        self.training_labels = training_labels
        
    def set_extracted_features(self):
        """
            Run feature extraction training images defined in self.training_images.

            For more details on feature extraction, see the FeatureExtractor module.

        """
        try:
            self.extracted_features = [FeatureExtractor.extract_features(image) for image in self.training_images]
        except AttributeError:
            print("Training images need to be set before feature extraction. Call set_training_data to initialize training data.")


    def cross_validate(self, 
             n_folds: int = 10,
             scoring: list = ["accuracy", "precision", "recall", "roc_auc"], 
            **kwargs):
        """
            Run cross validation on the model with its training data and labels. Store results in a cv_results member.

            --------
            
            Parameters:

            scoring (list): List of sklearn cross validation scoring identifiers. 
            Default: ["accuracy", "precision", "recall", "roc_auc"]. Assumes binary classification.
            Valid IDs are SKLearn classification identifiers.
            https://scikit-learn.org/stable/modules/model_evaluation.html
            n_folds (int): Number of cross validation folds. Default 10.

        """

        if "scoring" in kwargs:
            self.scoring = kwargs["scoring"]
        else: 
            self.scoring = scoring

        # Set stratified k folds using n_folds parameters
        skf = StratifiedKFold(n_folds)

        # If no extracted features exist, set them for the sklearn model
        if self.model_type == Model.SKLEARN:   
            self.set_extracted_features()
            try: 
                self.cv_results = cross_validate(self.model, self.extracted_features, self.training_labels, scoring = scoring, cv=skf)
            except AttributeError:
                print("Training data is not initialized. Call set_training_data to initialize training images and labels.")
        elif self.model_type == Model.PYTORCH: # pragma: no cover
            # TODO: Implement pytorch cross validation
            raise  Exception("Invalid model specified in marsvision.pipeline.Model")


    def write_cv_results(self, output_path: str = "cv_test_results.txt"):
        """
            Save cross validation results to a file. Shows results for individual folds, and the mean result for all folds,
            for all user specified classification metrics.

            -----
            Parameters:
            output_path (str): Path and file to write the results to.

        """
        output_file = open(output_path, "w")
        for score in self.scoring:
            cv_score_mean = np.mean(self.cv_results["test_" + score])
            output_file.write(score + "(all folds): " + str(self.cv_results["test_" + score]) + "\n")
            output_file.write(score + "(mean): " + str(cv_score_mean) + "\n")

        
    def train_model(self): # pragma: no cover
        """
            Trains a classifier using this object's configuration, as specified in the construct. 
            
            Either an SKLearn or Pytorch model will be trained on this object's data. The SKLearn model will be trained on extracted image features as specified in the FeatureExtractor module. The Pytorch model will be trained by running a CNN on the image data.
        """

        # Todo: Implement pytorch training
        if self.model_type == Model.PYTORCH:
            raise Exception("Invalid model specified in marsvision.pipeline.Model")
        elif self.model_type == Model.SKLEARN:
            # Extract features from every image in the batch,
            # then fit the sklearn model to these features.
            if self.extracted_features is None:
                self.set_extracted_features()
            self.model.fit(self.extracted_features, self.training_labels)
        else:
            raise Exception("No model specified in marsvision.pipeline.Model")


    def train_model_pytorchcnn(self, model, criterion, optimizer, scheduler, num_epochs = 25):

        # Set up simple dataset with labels and image data
        dataset = TensorDataset(Tensor(self.training_images), Tensor(self.training_labels))
        dataset_size  = len(dataset)

         # Train/Val/Test: 80/5/15
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [dataset_size * .8, dataset_size * .05, dataset_size * .15])
        dataloaders = {
            "train": torch.utils.data.DataLoader(train_dataset, batch_size = 4),
            "eval": torch.utils.data.DataLoader(val_dataset, batch_size = 4),
            "test": torch.utils.data.DataLoader(test_dataset, batch_size = 4)
        }

        # Parallelize if GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(num_epochs):
            print("Epoch: {}{}".format(epoch, num_epochs - 1))
            print("-") * 10
            
            # Train/Val/Test: 80/5/15
            for phase in ["train", "eval"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)


    def save_model(self, out_path: str = "model.p"):
        """
            Saves a pickle file containing this object's model.

            -------

            Parameters:
            out_path (str): The output location for the file.
        """

        if self.model_type == Model.SKLEARN:
            with open(out_path, "wb") as out_file:
                pickle.dump(self.model, out_file)
                
        if self.model_type == Model.PYTORCH: # pragma: no cover
            torch.save(self.model, out_path)


    def load_model(self, input_path: str, model_type: str):
        """
            Loads a model into this object from a pickle file, into the self.model member.

            -------

            Parameters:
            out_path(str): The input location of the file to be read.
            model_type(str): The model type. Either "sklearn" or "pytorch".
        """
        with open(input_path, 'rb') as in_file:
            self.model = pickle.load(in_file)
        self.model_type = model_type