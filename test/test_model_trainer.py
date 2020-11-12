from sklearn.linear_model import LogisticRegression as LR
from marsvision.pipeline.ConvNet import ConvNet
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
        self.training_images = dataloader.images
        self.labels =  dataloader.labels
    
    
    def test_save_load_pytorch(self):
        pytorch_nn = ConvNet()
        trainer_pytorch = ModelTrainer(self.training_images, self.labels, pytorch_nn, "pytorch")
        trainer_pytorch.train_model()

        nn_before = trainer_pytorch.model
        trainer_pytorch.save_model("test_pytorch_model.p")
        trainer_sklearn.load_model("test_pytorch_model.p")
        nn_after = trainer_pytorch.model

        for p1, p2 in zip(nn_before.parameters(), nn_after.parameters()):
            if p1.data.ne(p2.data).sum > 0:
                return False
        return True
        

    def test_save_load_lr(self):
        # Save and load the model, assert equal on its attributes
        sklearn_logistic_regression = LR()
        trainer_sklearn = ModelTrainer(self.training_images, self.labels, sklearn_logistic_regression, "sklearn")
        trainer_sklearn.train_model()

        lr_before = trainer_sklearn.model
        trainer_sklearn.save_model("test_lr_model.p")

        trainer_sklearn.load_model("test_lr_model.p")
        lr_after = trainer_sklearn.model

        os.remove("test_lr_model.p")
        test_lr_model = trainer_sklearn.model

        np.testing.assert_equal(lr_before.classes_, lr_after.classes_)
        np.testing.assert_equal(lr_before.coef_, lr_after.coef_)
        np.testing.assert_equal(lr_before.intercept_, lr_after.intercept_)
        np.testing.assert_equal(lr_before.n_iter_, lr_after.n_iter_)



    