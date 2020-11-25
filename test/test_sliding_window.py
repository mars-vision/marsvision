from marsvision.pipeline import sliding_window_predict
from marsvision.pipeline import Model
from unittest import TestCase
import cv2
import os
from numpy.testing import assert_array_equal

class TestSlidingWindow(TestCase):

    def test_sliding_window(self):
        current_dir = os.path.dirname(__file__)
        test_model_path = os.path.join(current_dir, "testing_models", "test_lr_model.p")
        test_image_path = os.path.join(current_dir, "marsface.jpg")

        test_model = Model()
        test_model.load_model(test_model_path, "sklearn")
        test_image = cv2.imread(test_image_path)

        test_stride_x = 32
        test_stride_y = 32
        test_window_dimension = 32

        expected_window_data = []
        for y in range(0, test_image.shape[0], test_stride_x):
            for x in range(0, test_image.shape[1], test_stride_y):
                # Store window attributes in a dictionary structure
                current_window_dictionary = {}
                
                # Slice window either to edge of image, or to end of window
                y_slice = min(test_image.shape[0] - y, test_window_dimension)
                x_slice = min(test_image.shape[1] - x, test_window_dimension)
                window = test_image[y:y_slice + y + 1, x:x_slice + x + 1]
                
                # Predict with model, store image coordinates of window
                current_window_dictionary["prediction"] = test_model.predict(window)
                current_window_dictionary["coordinates"] = (y, y_slice + y, x, x_slice + x)
                expected_window_data.append(current_window_dictionary)

        test_window_prediction_data = sliding_window_predict(test_image, test_model, test_window_dimension, test_window_dimension, test_stride_x, test_stride_y)

        assert_array_equal(test_window_prediction_data, expected_window_data)
