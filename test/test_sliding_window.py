from marsvision.pipeline import SlidingWindow
from marsvision.pipeline import Model
from unittest import TestCase
import cv2
import os
class TestSlidingWindow(TestCase):
    
    def test_sliding_window_predict(self):
        # Set up database file to test against if one does not exist
        self.model = Model()
        test_file_path = os.path.join(os.path.dirname(__file__), "test_files")
        test_model_path = os.path.join(test_file_path, "test_lr_model.p")
        self.model.load_model(test_model_path, "sklearn")
        self.test_image = cv2.imread(os.path.join(test_file_path, "marsface.jpg"))
        sw_test = SlidingWindow(self.model, os.path.join(test_file_path, "marsvision.db"))
        sw_test.sliding_window_predict(self.test_image)
        


