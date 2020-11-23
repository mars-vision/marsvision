import numpy as np
import sklearn
import os
import torch
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from marsvision.pipeline import FeatureExtractor
import torch.nn as nn
from sklearn.model_selection import cross_validate, StratifiedKFold

class Model:
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"

    def __init__(self, 
        model = None,
        model_type: str = PYTORCH,
        **kwargs):
        """
            Model class that serves as an abstract wrapper for either an sklearn or pytorch model.

            Contains methods for making predictions, and for cross validating the model and writing results to a file.

            --------
            
            Parameters:

            training_images (numpy.ndarray): Batch of images to train on 
            training_labels: Class labels for the training images
            model: Either an sklearn machine learning model, or a pytorch neural network.
            model_type (str): String identifier for the type of model. Determines how the model will be trained in this class.
        """

        if "training_images" in kwargs:
            self.training_images = kwargs["training_images"]

        if "training_labels" in kwargs:
            self.training_labels = kwargs["training_labels"]

        if "num_epochs" in kwargs:
            self.num_epochs = kwargs["num_epochs"]

        self.model_type = model_type
        self.model = model
        
        # Initialize extracted features to none; use this member when we use the sklearn model
        self.extracted_features = None

    def predict(self, image: np.ndarray):
        """
            Return a prediction from the trained model in this object.
        """
        if self.model_type == Model.SKLEARN:
            image_features = np.array(FeatureExtractor.extract_features(image))
            return self.model.predict(image_features.reshape(1, -1))
        elif self.model_type == Model.PYTORCH:
            # TODO: Implement pytorch for model class
            pass
        else:
            raise Exception("No model specified in marsvision.pipeline.Model")


    def set_extracted_features(self):
        self.extracted_features = [FeatureExtractor.extract_features(image) for image in self.training_images]

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
            if self.extracted_features is None:
                self.set_extracted_features()
            self.cv_results = cross_validate(self.model, self.extracted_features, self.training_labels, scoring = scoring, cv=skf)
        elif self.model_type == Model.PYTORCH:
            # TODO: Implement pytorch cross validation
            pass
        else:
            raise Exception("No model type specified in marsvision.pipeline.Model")

    def write_cv_results(self, output_path: str = "cv_test_results.txt"):
        """
            Save cross validation results to a file. Shows results for individual folds, and the mean result for all folds,
            for all user specified classification metrics.

            -----
            Parameters:
            output_path (str): Path and file to write the results to.

        """
        if self.cv_results is None:
            raise Exception("No cross validation results to write. Call Model.cross_validate first.")

        output_file = open(output_path, "w")
        for score in self.scoring:
            cv_score_mean = np.mean(self.cv_results["test_" + score])
            output_file.write(score + "(all folds): " + str(self.cv_results["test_" + score]) + "\n")
            output_file.write(score + "(mean): " + str(cv_score_mean) + "\n")

        
    def train_model(self):
        """
            Trains a classifier using this object's configuration, as specified in the construct. 
            
            Either an SKLearn or Pytorch model will be trained on this object's data.

            The SKLearn model will be trained on extracted image features as specified in the FeatureExtractor module.

            The Pytorch model will be trained by running a CNN on the image data.
        """

        # Todo: Implement pytorch training
        #if self.model_type == Model.PYTORCH:
            #self.train_pytorch()

        if self.model_type == Model.SKLEARN:
            # Extract features from every image in the batch,
            # then fit the sklearn model to these features.
            if self.extracted_features is None:
                self.set_extracted_features()
            self.model.fit(self.extracted_features, self.training_labels)
        else:
            raise Exception("No model specified in marsvision.pipeline.Model")

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
                
        if self.model_type == Model.PYTORCH:
            torch.save(model, out_filename)


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