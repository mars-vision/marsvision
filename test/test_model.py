from sklearn.linear_model import LogisticRegression as LR
from marsvision.pipeline.ConvNet import ConvNet
import pickle
from marsvision.pipeline.Model import Model
from marsvision.utilities import DataLoader
from unittest import TestCase
import os
import numpy as np

class TestModel(TestCase):
    def setUp(self):
        # Run the dataloader to get a dataframe of image features and classes
        # So we can pass data into our model trainer
        self.current_dir = os.path.dirname(__file__)
        test_image_path = os.path.join(self.current_dir, "test_data")

        # Load test images into memory using DataLoader utility
        self.dataloader = DataLoader(test_image_path, test_image_path)
        self.dataloader.data_reader()

        self.training_images = self.dataloader.images
        self.labels = [1 if name == "dust" else 0 for name in self.dataloader.labels]
        self.sklearn_logistic_regression = LR()
        self.model_sklearn = Model(self.sklearn_logistic_regression, "sklearn", training_images = self.training_images, training_labels = self.labels)
        self.model_sklearn.train_model()


    def test_save_load_inference(self):
        # Make a prediction on an image, 
        # save the model,
        # load the same with a new instance of Model from the file,
        # predict with the new instance,
        # assert equal
        predict_image = self.dataloader.images[0]
        expected_prediction = self.model_sklearn.predict(predict_image)
        self.model_sklearn.save_model("model.p")
        test_model = Model()
        test_model.load_model("model.p", "sklearn")
        os.remove("model.p")
        test_prediction = test_model.predict(predict_image)
        self.assertEqual(test_prediction[0], expected_prediction[0])
        
    def test_write_cv_results(self):
        # Run model's cross validation, save file
        results_path = os.path.join(self.current_dir, "cv_test_results.txt")
        self.model_sklearn.cross_validate(2)
        self.model_sklearn.write_cv_results(results_path)
        os.remove(results_path)
        # An assertion would be nice here

    def test_save_load_lr(self):
        # Save and load the model, assert equal on its attributes
        lr_before = self.model_sklearn.model
        self.model_sklearn.save_model("test_lr_model.p")

        self.model_sklearn.load_model("test_lr_model.p", "sklearn")
        lr_after = self.model_sklearn.model

        os.remove(os.path.join(os.getcwd(), "test_lr_model.p"))
        test_lr_model = self.model_sklearn.model

        np.testing.assert_equal(lr_before.classes_, lr_after.classes_)
        np.testing.assert_equal(lr_before.coef_, lr_after.coef_)
        np.testing.assert_equal(lr_before.intercept_, lr_after.intercept_)
        np.testing.assert_equal(lr_before.n_iter_, lr_after.n_iter_)

    """
    Come back to this when we come back to pytorch
    def test_save_load_pytorch(self):
        pytorch_nn = ConvNet()
        trainer_pytorch = Model(self.training_images, self.labels, pytorch_nn, "pytorch")
        trainer_pytorch.train_model()

        nn_before = trainer_pytorch.model
        trainer_pytorch.save_model("test_pytorch_model.p")
        self.self.model_sklearn.load_model("test_pytorch_model.p")
        nn_after = trainer_pytorch.model

        for p1, p2 in zip(nn_before.parameters(), nn_after.parameters()):
            if p1.data.ne(p2.data).sum > 0:
                return False
            eturn True
    """ 
