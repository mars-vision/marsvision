from sklearn.linear_model import LogisticRegression as LR
import pickle
from marsvision.pipeline.ModelTrainer import ModelTrainer
from marsvision.utilities import DataLoader
from unittest import TestCase
import os
import numpy as np

class TestModelTrainer(TestCase):
    ## Question: best practices for unit testing machine learning models

    def setUp(self):
        # Run the dataloader to get a dataframe of image features and classes
        # So we can pass data into our model trainer
        current_dir = os.path.dirname(__file__)
        test_image_path = os.path.join(current_dir, "test_images_loader")
        test_model_path = os.path.join(current_dir, "testing_models", "test_lr_model.p")

        dataloader = DataLoader(test_image_path, test_image_path)
        dataloader.data_reader()
        dataloader.data_transformer()

        data = dataloader.df
        training_data = data.iloc[:,0:5]
        labels = data["class_code"]

        print(training_data)
        sklearn_logistic_regression = LR()

        self.trainer_sklearn = ModelTrainer(training_data, labels, sklearn_logistic_regression, "sklearn")
        self.trainer_sklearn.train_model()


    def test_save_load(self):
        # Save and load the model, assert equal on its attributes
        
        lr_before = self.trainer_sklearn.model
        self.trainer_sklearn.save_model("test_lr_model.p")

        self.trainer_sklearn.load_model("test_lr_model.p")
        lr_after = self.trainer_sklearn.model

        os.remove("test_lr_model.p")
        test_lr_model = self.trainer_sklearn.model

        np.testing.assert_equal(lr_before.classes_, lr_after.classes_)
        np.testing.assert_equal(lr_before.coef_, lr_after.coef_)
        np.testing.assert_equal(lr_before.intercept_, lr_after.intercept_)
        np.testing.assert_equal(lr_before.n_iter_, lr_after.n_iter_)



    