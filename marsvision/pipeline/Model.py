import numpy as np
import cv2
import sklearn
import os
import torch
import pickle
from marsvision.pipeline.FeatureExtractor import *
from marsvision.vision import DeepMarsDataset
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import plot_roc_curve
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

        if self.model_type == Model.PYTORCH:
            assert(self.dataset_root_directory is not None), "No dataset directory specified. You must specify a dataset root directory when using a Pytorch model."

        if type(model) == str:
            self.load_model(model, self.model_type)
        else:
            self.model = model
        
        # Initialize extracted features to none; use this member when we use the sklearn model
        self.extracted_features = None

    def predict(self, image_list: np.ndarray, input_dimension: int = None):
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
            # Rescale the image according to the input dimension specified by the user.
            # Apply the standard normalization expected by pre-trained models.
            # Is there a way to parallelize this procedure?
            normalize = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = torch.empty(size=(len(image_list), 3, input_dimension, input_dimension))
            for i in range(len(image_list)):
                img_tensor = Tensor(
                    cv2.resize(image_list[i], (input_dimension, input_dimension), interpolation = cv2.INTER_AREA)
                ).transpose(0, 2)
                input_tensor[i] = normalize(img_tensor)

            # Output index of the maximum confidence score per sample.
            return self.model(input_tensor).argmax(dim=1)
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


    def train_model_pytorchcnn(self):
        """
            This is an internal helper function which handles the training of a pytorch CNN model.
 
            The various hyperparameters for CNN training, such as learning rate and number of epochs, can be found in the package's config file.

            ----

            Parameters:

            root_dir (str): Path to the Deep Mars dataset.

        """
        # Handle Pytorch configuartion file parameters here.
        # Extracting these to named variables because
        # We can later add conditionals that use kwargs with the same keys,
        # to make these function calls a bit more customizable to the user.
        
        pytorch_parameters = self.config["pytorch_cnn_parameters"]
        num_epochs = pytorch_parameters["num_epochs"]
        learning_rate = pytorch_parameters["gradient_descent_learning_rate"]
        momentum = pytorch_parameters["gradient_descent_momentum"]
        step_size = pytorch_parameters["scheduler_step_size"]
        gamma = pytorch_parameters["scheduler_gamma"]
        train_proportion = pytorch_parameters["train_proportion"]
        test_proportion = pytorch_parameters["test_proportion"]
        root_dir = self.dataset_root_directory

        # Initialize using values from the config file.
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        # Decay by a factor of gamma every step_size epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        criterion = nn.CrossEntropyLoss() 

        # Instantiate the dataset using our custom DeepMarsDataset class (found in the vision folder)
        dataset = DeepMarsDataset(root_dir)
        dataset_size  = len(dataset)

        # Determine the number of samples for our different sets as ints.
        num_train_samples = int(dataset_size * train_proportion)
        num_val_samples = int(dataset_size * test_proportion)
        num_test_samples = dataset_size - num_train_samples - num_val_samples
        data_sizes = {
            "train": num_train_samples,
            "val": num_val_samples,
            "test": num_test_samples
        }

        # Split the dataset using the above values.
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, 
            [
                data_sizes["train"],
                data_sizes["val"],
                data_sizes["test"]
            ],
            generator = torch.Generator().manual_seed(42)
        )

        # Finally, instantiate the dataloaders using the split sets.
        DataLoaders = {
            "train": torch.utils.data.DataLoader(train_dataset, batch_size = 4),
            "val": torch.utils.data.DataLoader(val_dataset, batch_size = 4),
            "test": torch.utils.data.DataLoader(test_dataset, batch_size = 4)
        }

        # Parallelize if a valid GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Training starts here.
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            
            # Train/Val/Test: 80/5/15
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0

                # Get samples in batches (specified in the DataLoader objects).
                for sample in DataLoaders[phase]:
                    inputs = Tensor(sample["image"]).to(device)
                    labels = sample["label"]

                    # Zero the gradients before the forward pass
                    optimizer.zero_grad()

                    # Forward pass if in train phase
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            # Gradient descent / adjust weights
                            loss.backward()
                            optimizer.step()
                            
                    # Note -- what's happening in this loss calculation?
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += int(torch.sum(preds == labels))
                    print("Running loss: {} | Running corrects: {}".format(
                    running_loss, running_corrects))

                    if phase == "train":
                        scheduler.step()

                epoch_loss = running_loss / data_sizes[phase]
                epoch_acc = running_corrects / data_sizes[phase]
                print(data_sizes[phase])
                print('{} loss: {:.4f} Acc: {:.4f} | Images trained on: {}'.format(
                    phase, epoch_loss, epoch_acc, data_sizes[phase]))

                # In the eval phase, get the accuracy for this epoch
                # If the model's current state is better than the best model seen so far,
                # replace the best model weights
                # with the previous best model weights on previous epochs
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    
        print('Best Epoch Acc: {:.4f}'.format(best_acc))
        self.model.load_state_dict(best_model_wts)

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

        