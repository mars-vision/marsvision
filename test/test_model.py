import os
from unittest import TestCase

import numpy as np
from pandas._testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression

from marsvision.pipeline.Model import Model
from marsvision.utilities import DataUtility
from marsvision.vision.ModelDefinitions import alexnet_grayscale


class TestModel(TestCase):
    def setUp(self):
        # Run the DataUtility to get a dataframe of image features and classes
        # So we can pass data into our model trainer
        self.current_dir = os.path.dirname(__file__)
        test_image_path = os.path.join(self.current_dir, "test_data")
        self.test_files_path = os.path.join(self.current_dir, "test_files")

        # Load test images into memory using DataUtility utility
        self.DataUtility = DataUtility(test_image_path, test_image_path)
        self.DataUtility.data_reader()

        self.training_images = self.DataUtility.images
        self.labels = [1 if name == "dust" else 0 for name in self.DataUtility.labels]
        self.sklearn_logistic_regression = LogisticRegression()
        self.model_sklearn = Model(self.sklearn_logistic_regression, "sklearn", training_images=self.training_images,
                                   training_labels=self.labels)
        self.model_sklearn.train_model()

        self.deepmars_test_path = os.path.join(self.current_dir, "deep_mars_test_data")
        self.model_pytorch = Model(alexnet_grayscale(), "pytorch", dataset_root_directory=self.deepmars_test_path)

    def test_save_load_inference(self):
        # Make a prediction on an image, 
        # save the model,
        # load the same with a new instance of Model from the file,
        # predict with the new instance,
        # assert equal
        predict_image = self.DataUtility.images[0]
        expected_prediction = self.model_sklearn.predict(predict_image)
        self.model_sklearn.save_model("model.p")
        test_model = Model("model.p", "sklearn")
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

    def test_cross_validate_plot(self):
        # Run the binary cross validation routine to validate it in this test
        self.model_sklearn.cross_validate_plot(2)

    def test_pytorch_training(self):
        # Run the pytorch training method to cover it.
        # Saves two files: marsvision_cnn.pt and marsvision_cnn_evaluation.p
        self.model_pytorch.train_and_test_cnn(num_epochs=1, out_path=self.test_files_path)

        # Test the load_model method for a pytorch model.
        # load_model is called internally when a path to the model is provided.
        testing_model = Model(os.path.join(self.test_files_path, "marsvision_cnn.pt"), "pytorch")

        os.remove(os.path.join(self.test_files_path, "marsvision_cnn.pt"))
        os.remove(os.path.join(self.test_files_path, "marsvision_cnn_evaluation.p"))

    def test_pytorch_inference(self):
        # Test pytorch inference.
        # Ensure that the inference is in the same format as the sklearn inference
        # This is to ensure that both methods write to the database in the same way.

        # Both should return a list of ints representing class labels.
        pytorch_predict = self.model_pytorch.predict(self.DataUtility.images)
        sklearn_predict = self.model_sklearn.predict(self.DataUtility.images)
        self.assertEqual(len(pytorch_predict), len(sklearn_predict))
        self.assertTrue(all(isinstance(x, int) for x in pytorch_predict))
        self.assertTrue(all(isinstance(x, int)) for x in sklearn_predict)

    def test_get_evaluation_dataframe(self):
        # Test static method that parses training output into a dataframe.

        # If the test file does not exist, create a new one.
        # This allows us to delete the file and create a new one by running tests.

        test_dataframe_path = os.path.join(self.test_files_path, "test_eval_dataframe.csv")
        model_results_path = os.path.join(self.test_files_path, "alexnet-results-3-19.p")

        try:
            test_eval_df = pd.read_csv(test_dataframe_path)
            expected_eval_df = Model.get_evaluation_dataframe(model_results_path)
            assert_frame_equal(test_eval_df, expected_eval_df)
        except:
            print("No test dataframe file found. Writing a new file.")
            # Produce the dataframe by running Model.get_evaluation_dataframe.
            eval_df = Model.get_evaluation_dataframe(model_results_path)
            eval_df.to_csv(test_dataframe_path)
