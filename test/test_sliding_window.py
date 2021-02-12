from marsvision.pipeline import SlidingWindow
from marsvision.pipeline import Model
from marsvision.utilities import DataLoader
from unittest import TestCase
import cv2
import os
import pandas as pd
import sqlite3
import numpy as np

class TestSlidingWindow(TestCase):
    
    def test_sliding_window_predict(self):
        model = Model()
        test_file_path = os.path.join(os.path.dirname(__file__), "test_files")
        test_model_path = os.path.join(test_file_path, "test_lr_model.p")


        # Get the images in test_data as a batch
        model.load_model(test_model_path, "sklearn")
        loader = DataLoader("test_data")
        loader.data_reader()
        test_images = np.array(loader.images)
        test_filenames = loader.file_names

        # Testing is done by writing a new DB file 
        # and comparing it with an "expected" database file
        test_db_path = os.path.join(test_file_path, "marsvision.db")
        expected_db_path = os.path.join(test_file_path, "marsvision_expected.db")

        # Ensure there is no test db file
        try:
            os.remove(test_db_path)
        except OSError:
            pass
        
        # If no expected db file is present, create one.
        # When we intentionally want to change database output,
        # we can delete the marsvision_expected file and run tests to create a new one.
        sw_test = SlidingWindow(model)
        if not os.path.isfile(expected_db_path):
            sw_test.db_path = expected_db_path
            sw_test.sliding_window_predict(test_images, test_filenames)
        sw_test.db_path = test_db_path
        sw_test.sliding_window_predict(test_images, test_filenames)

        test_conn = sqlite3.connect(test_db_path)
        expected_conn = sqlite3.connect(expected_db_path)
        

        # Asserting tables in both databases are the same
        test_global_table = pd.read_sql("SELECT * FROM global", test_conn)
        expected_global_table = pd.read_sql("SELECT * FROM global", expected_conn)
        
        test_image_table = pd.read_sql("SELECT * FROM windows", test_conn)
        expected_image_table = pd.read_sql("SELECT * FROM windows", expected_conn)

        # Remove output file; otherwise it gets appended to every time we run tests
        test_conn.close()
        expected_conn.close()
        os.remove(test_db_path)
        pd.testing.assert_frame_equal(test_image_table, expected_image_table)
        pd.testing.assert_frame_equal(test_global_table, expected_global_table)


