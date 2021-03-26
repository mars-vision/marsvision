import numpy as np
import cv2
from PIL import Image
import sklearn
import os
import torch
import pickle
import pandas as pd
from marsvision.pipeline.FeatureExtractor import *
from marsvision.vision import DeepMarsDataset
from sklearn.model_selection import cross_validate, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import plot_roc_curve, multilabel_confusion_matrix
from typing import List
import torch
import torchvision
from torchvision import transforms
from torch import Tensor
from torch import nn
from torch import optim
import copy
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from marsvision.config_path import CONFIG_PATH
import yaml

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

            model: Either an sklearn machine learning model, or a pytorch neural network. Can be a path to a file or a model object.

            model_type (str): String identifier for the type of model. Determines how the model will be trained in this class.


            **kwargs:
            training_images (numpy.ndarray): Batch of images to train on 
            training_labels: Class labels for the training images
            dataset_root_directory: The root directory of the Deep Mars dataset to train on.
        """

        # Open config file
        with open(CONFIG_PATH) as yaml_cfg:
            self.config = yaml.load(yaml_cfg)

        if "training_images" in kwargs:
            self.training_images = kwargs["training_images"]

        if "training_labels" in kwargs:
            self.training_labels = kwargs["training_labels"]

        if "dataset_root_directory" in kwargs:
            self.dataset_root_directory = kwargs["dataset_root_directory"]

        self.model_type = model_type

        if type(model) == str:
            self.load_model(model, self.model_type)
        else:
            self.model = model
        
        # Initialize extracted features to none; use this member when we use the sklearn model
        self.extracted_features = None

    def predict(self, image_list: np.ndarray, input_dimension: int = None, transform = None):
        """
            Run inference using self.model on a list of images using the currently instantiated model.
            
            This model can either be an sklearn model or a pytorch model.

            Returns a list of inferences.
            
            ---

            Parameters:
            image_list (List[np.ndarray]): Batch of images to run inference on with this model.
            crop_size: (Tuple[int, int]): Tuple containing width and height of images if they need to be cropped.
        """

        image_list = np.array(image_list)
        # Handle the case of a single image by casting it as a list with itself in it
        if len(image_list.shape) == 3:
            image_list = [image_list]

        if self.model_type == Model.SKLEARN:
            # Iterate images in batch,
            # Extract features,
            # Use self.model for inference on each
            # Return a list of inferences
            image_list = np.array(image_list)
            image_feature_list = []
            for image in image_list:
                image_feature_list.append(FeatureExtractor.extract_features(image))
            inference_list = self.model.predict(image_feature_list)
            return list(map(int, inference_list))
        elif self.model_type == Model.PYTORCH:
            if input_dimension is None:
                config_pytorch = self.config["pytorch_cnn_parameters"]
                input_dimension = config_pytorch["input_dimension"]
                crop_dimension = config_pytorch["crop_dimension"]

            # Rescale the image according to the input dimension specified by the user.
            # Apply the standard normalization expected by pre-trained models.
            transform = transforms.Compose([
                transforms.Resize(crop_dimension),
                transforms.CenterCrop(input_dimension),
                transforms.ToTensor(), # normalize to [0, 1]
                transforms.Normalize(
                    mean=[0.485],
                    std=[0.229],
                ),
            ])

            # Since Deep Mars is trained on grayscale images, transform the images to greyscale.
            input_tensor = torch.empty(size=(len(image_list), 1, input_dimension, input_dimension))
            for i in range(len(image_list)):
                img = cv2.cvtColor(image_list[i], cv2.COLOR_RGB2GRAY)
                img_pil = Image.fromarray(img)
                input_tensor[i] = transform(img_pil)

            # Output index of the maximum confidence score per sample.
            # Return the output tensor as a list.
            return self.model(input_tensor).argmax(dim=1).tolist()
        else:
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

    
    def cross_validate_plot(self, title: str = "Binary Cross Validation Results", n_folds: int = 2):
        """
            Run cross validation on a binary classification problem,
            and make a matplotlib plot of the results.

            ---
            
            Parameters

            Title(str): Title of the figure.
            n_folds(int): Number of folds.
        """

        fig, ax = plt.subplots()

        cv_results = self.cross_validate_binary_metrics(n_folds, ax)

        fig.set_size_inches(18.5, 10.5)
            
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=title)
            
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Random Chance', alpha=.8)
            
        ax.plot(np.linspace(0, 1, 100), cv_results["mean_tpr"], color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (cv_results["mean_auc"], 
                                                            cv_results["std_auc"]),
                lw=2, alpha=.8)
            
        tprs_upper = np.minimum(cv_results["mean_tpr"] + cv_results["std_tpr"], 1)
        tprs_lower = np.maximum(cv_results["mean_tpr"] - cv_results["std_tpr"], 0)
        ax.fill_between(np.linspace(0, 1, 100), tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        ax.legend()
        plt.show()

        return cv_results


    def cross_validate_binary_metrics(self, 
             n_folds: int = 5,
             ax = None):
        """
            Run cross validation on a binary classification problem on the basic pipeline, and return the results as a dictionary.

            This is mainly a helper function for the cross_validate_plot function, which cross validates and plotes ROC curves for each fold.

            This method returns the domain over which the plot is constructed as well as the tpr and fpr values, alongside standard binary classification measures for each fold: precision, recall, accuracy, auc.

            This method assumes that there are only two labels in the training label member of this class.

            --------
            
            Parameters:

            n_dolfds (int): Number of folds.
            ax: Matplotlib axis on which to show the plot.

        """

        # Set stratified k folds using n_folds 
        stratified_kfold = StratifiedKFold(n_folds)
        try: 
            self.set_extracted_features()
            x = np.array(self.extracted_features)
            y = np.array(self.training_labels)
        except AttributeError:
            print("Training data is not initialized. Call set_training_data to initialize training images and labels.")
            
        precisions = []
        aucs = []
        accs = []
        recalls = []
        visualizations = []
        tprs = []
        x_domain = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(stratified_kfold.split(self.extracted_features, self.training_labels)):
            self.model.fit(x[train], y[train])
            viz = plot_roc_curve(self.model, x[test], y[test])
            plt.close()
            visualizations.append(plot_roc_curve(self.model, x[test], y[test], ax = ax))
            interp_tpr = np.interp(x_domain, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            y_predict = np.array(self.model.predict(x[test]))
            y_test = np.array(y[test])
            false_negatives = np.sum(y_test[y_predict == 0] != y_predict[y_predict == 0])
            true_positives = np.sum(y_test[y_predict == 1] == y_predict[y_predict == 1])
            false_positives = np.sum(y_test[y_predict == 1] != y_predict[y_predict == 1])

            precisions.append(true_positives / (true_positives + false_positives))
            recalls.append(true_positives / (false_negatives + true_positives))
            accs.append(np.sum(y_predict == y[test]) / len(y[test]))
                    
        return {
            "precisions": precisions,
            "recalls": recalls,
            "roc_aucs": aucs,
            "mean_auc": np.mean(aucs),
            "std_auc": np.std(aucs),
            "accuracies": accs,
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "tprs": tprs,
            "mean_tpr": np.mean(tprs, axis=0),
            "std_tpr": np.std(tprs),
            "x_domain": x_domain
        }


    def cross_validate(self, 
             n_folds: int = 10,
             scoring: list = ["accuracy", "precision", "recall", "roc_auc"], 
            **kwargs):
        """
            Run cross validation on the model with its training data and labels. Return the results.

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

        
    def train_model(self):
        """
            Trains a classifier using this object's configuration, as specified in the constructor. 
            
            Either an SKLearn or Pytorch model will be trained on this object's data. The SKLearn model will be trained on extracted image features as specified in the FeatureExtractor module. The Pytorch model will be trained by running a CNN on the image data.
        
            -----
            
            Parameters:

            root_dir (str): Root directory of the Deep Mars dataset.
        """

        if self.model_type == Model.PYTORCH:   
            self.train_model_pytorchcnn()
        elif self.model_type == Model.SKLEARN:
            # Extract features from every image in the batch,
            # then fit the sklearn model to these features.
            if self.extracted_features is None:
                self.set_extracted_features()
            self.model.fit(self.extracted_features, self.training_labels)
        else:
            raise Exception("No model specified in marsvision.pipeline.Model")


    def train_and_test_cnn(self, out_path: str = None, num_epochs: int = None, test_proportion: float = None):
        """
            Train and evaluate a pytorch CNN model.

            The model is defined in this class' constructor. If a valid pytorch model is used for this object, use this method to train a model and save evaluation information to a file. The trained model will be saved to a file path specified by the user, or directly to the current working directly by default. The information is also retained in a member variable called cnn_evaluation_results. 

            The data is split into train and test sets that are stratified, i.e. the class distributions are preserved between training and testing sets.

            The evaluation data is contained in a dictionary method returns a dictionary containing these keys:

            epoch_acc: List of accuracies for all epochs.
            epoch_loss: List of loss for all epochs.
            predicted_labels: 2d array. Each row is a list of predicted labels for the epoch corresponding to its index.
            ground_truth_labels: True labels over the dataset. 
            prediction_probabilities: Maximum probabilities of predictions.

            These dictionary members are all lists whose indices correspond to training epochs.

            The various hyperparameters for CNN training, such as learning rate and number of epochs, can be found in the config file.

            ----

            Parameters:

            root_dir (str): Path to the Deep Mars dataset.
            out_path: (str): The directory to which files should be saved.
            num_epochs (int): Named parameter. Number of training epochs. Default values are located in the config file.
            test_proportion (float): Named parameter. Proportion of data to be used for model evaluation. Expected to be a value in range [0, 1]. The complement (1 - test_proportion) is used to train the model.


        """
        # Handle Pytorch configuartion file parameters here.
        # Extracting these to named variables because
        # We can later add conditionals that use kwargs with the same keys,
        # to make these function calls a bit more customizable to the user.
        pytorch_parameters = self.config["pytorch_cnn_parameters"]
        
        learning_rate = pytorch_parameters["gradient_descent_learning_rate"]
        momentum = pytorch_parameters["gradient_descent_momentum"]
        step_size = pytorch_parameters["scheduler_step_size"]
        gamma = pytorch_parameters["scheduler_gamma"]
        
        num_classes = pytorch_parameters["num_output_classes"]
        batch_size = pytorch_parameters["batch_size"]
        num_workers = pytorch_parameters["num_workers"]
        root_dir = self.dataset_root_directory
        
        # Use config file values if these are not present in kwargs
        if num_epochs is None:
            num_epochs = pytorch_parameters["num_epochs"]
        
        if test_proportion is None:
            test_proportion = pytorch_parameters["test_proportion"]

        # Parallelize if a valid GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)

        # Initialize using values from the config file.
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # Decay by a factor of gamma every step_size epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        criterion = nn.CrossEntropyLoss() 

        # Instantiate the dataset using our custom DeepMarsDataset class (found in the vision folder)
        dataset = DeepMarsDataset(root_dir)
        dataset_size = len(dataset)
        dataset_label_list = dataset.get_labels()

        # Generate a stratified train/test split using labels from the dataset object.
        stratified_shufflesplit = StratifiedShuffleSplit(n_splits=1, test_size=test_proportion)
        (train_idx, test_idx) = next(stratified_shufflesplit.split(np.zeros(dataset_size), dataset_label_list))
        

        # Determine the number of samples for our different sets
        num_train_samples = len(train_idx)
        num_test_samples = len(test_idx)
        data_sizes = {
            "train": num_train_samples,
            "test": num_test_samples
        }

        # These sampler objects get passed to the dataloader with our training and testing indices,
        # to split the data accordingly.
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        # Finally, instantiate the dataloaders using the samplers.
        DataLoaders = {
            "train": torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, sampler = train_sampler),
            "test": torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, sampler = test_sampler)
        }
       
        # Training starts here.
        # Since this run is for evaluation,
        # keep track of metrics here as well.
        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        # List of dictionaries indexed by epoch:
        # (predicted_labels, prediction_probabilities, ground_truth_labels)
        # For use in model evaluation
        epoch_metrics = {
            "epoch_acc":  [],
            "epoch_loss": [],
            "predicted_labels": [],
            "prediction_probabilities": [],
            "ground_truth_labels": []
        }

        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            # Train/Val
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                    # Create a new entry in the epoch metrics lists
                    # for each training phase every epoch, 
                    # containing test results.
                    epoch_metrics["predicted_labels"].append([])
                    epoch_metrics["prediction_probabilities"].append([])
                    epoch_metrics["ground_truth_labels"].append([])
                    
                    
                
                running_loss = 0.0
                running_corrects = 0

                # Get samples in batches
                for sample in DataLoaders[phase]:
                    inputs = sample["image"].to(device)
                    labels = sample["label"].to(device)

                    # Zero the gradients before the forward pass
                    optimizer.zero_grad()

                    # Forward pass. If in train phase, keep grad enabled.
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        scores, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            # Gradient descent / adjust weights
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += int(torch.sum(preds == labels))

                    if phase == "test":
                        epoch_metrics["predicted_labels"][epoch].extend(preds.flatten().tolist())
                        # Run a softmax function on the scores to turn them into
                        # probability scores in the range [0, 1].
                        scores_normalized = torch.nn.functional.softmax(scores)
                        epoch_metrics["prediction_probabilities"][epoch].extend(scores_normalized.flatten().tolist())
                        epoch_metrics["ground_truth_labels"][epoch].extend(labels.flatten().tolist())


                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / data_sizes[phase]
                epoch_acc = running_corrects / data_sizes[phase]

                if phase == "test":
                    epoch_metrics["epoch_loss"].append(epoch_loss)
                    epoch_metrics["epoch_acc"].append(epoch_acc)


                print('{} loss: {:.4f} Acc: {:.4f} | Images parsed: {}'.format(
                    phase, epoch_loss, epoch_acc, data_sizes[phase]))

                # In the eval phase, get the accuracy for this epoch
                # If the model's current state is better than the best model seen so far,
                # replace the best model weights
                # with the previous best model weights on previous epochs
                if phase == 'test' and best_acc < epoch_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    

        print('Best Epoch Acc: {:.4f}'.format(best_acc))
        self.model.load_state_dict(best_model_wts)
        self.cnn_evaluation_results = epoch_metrics

        if out_path is None:
            self.save_cnn_evaluation_results()
            self.save_model("marsvision_cnn.pt")
        else:
            self.save_cnn_evaluation_results(os.path.join(out_path,"marsvision_cnn_evaluation.p"))
            self.save_model(os.path.join(out_path, "marsvision_cnn.pt") )

    def save_cnn_evaluation_results(self, out_path: str = "marsvision_cnn_evaluation.p"):
        """ 
            Helper function that saves the evaluation results of CNN training.
        """
        with open(out_path, "wb") as out_file:
            pickle.dump(self.cnn_evaluation_results, out_file)


    def save_model(self, out_path: str = "model.p"):
        """
            Saves a pickle file containing this object's model. The model can either be a Pytorch or SKlearn model.

            -------

            Parameters:
            out_path (str): The output location for the file.
        """

        if self.model_type == Model.SKLEARN:
            with open(out_path, "wb") as out_file:
                pickle.dump(self.model, out_file)
        elif self.model_type == Model.PYTORCH: # pragma: no cover
            torch.save(self.model, out_path)
        else:
            raise Exception("No model type selected in this Model object.")


    def load_model(self, input_path: str, model_type: str):
        """
            Loads a model into this object from a file, into the self.model member. The model can either be a Pytorch model saved via torch.save() or an SKlearn model.

            -------

            Parameters:

            out_path(str): The input location of the file to be read.
            model_type(str): The model type. Either "sklearn" or "pytorch".
        """

        self.model_type = model_type

        if self.model_type == Model.SKLEARN:
            with open(input_path, 'rb') as in_file:
                self.model = pickle.load(in_file)
        elif self.model_type == Model.PYTORCH:
            self.model = torch.load(input_path)
        else:
            raise Exception("Model_type does not match a valid class. Specify 'pytorch' or 'sklearn'.")

        