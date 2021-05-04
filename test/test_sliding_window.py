from marsvision.pipeline import SlidingWindow
from marsvision.pipeline import Model
from marsvision.path_definitions import PDSC_TABLE_PATH
from unittest import TestCase
import cv2
import os
import pandas as pd
import sqlite3
import numpy as np
import pdsc
import requests

class TestSlidingWindow(TestCase):
    
    def test_sliding_window_predict(self):
        test_file_path = os.path.join(os.path.dirname(__file__), "test_files")
        test_model_path = os.path.join(test_file_path, "test_lr_model.p")
        test_data_path = os.path.join(os.path.dirname(__file__), "test_data/dust")
        model = Model(test_model_path, "sklearn")

        # Get the images in test_data as a batch
        model.load_model(test_model_path, "sklearn")
        
        # Get PDSC Metadata objects to pass into the sliding window object.
        client = pdsc.PdsClient(PDSC_TABLE_PATH)
        test_metadata_ids = [
            "PSP_006569_1135",
            "PSP_006570_1820",
            "PSP_006571_1905"
        ]

        metadata_list = client.query_by_observation_id("hirise_rdr", test_metadata_ids)
        metadata_list = [m for m in metadata_list if 'RED' in m.product_id]

        # Fill in image data from metadata
        image_list = []
        for metadata in metadata_list: 
            # Get path to map projected image by using .NOMAP.
            url_suffix = metadata.file_name_specification.split(".")[0] + ".NOMAP.browse.jpg"
            url = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/" + url_suffix
            response = requests.get(url, stream=True).raw
            image = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_list.append(image)

        # Testing is done by writing a new DB file 
        # and comparing it with an "expected" database file
        test_db_path = os.path.join(test_file_path, "marsvision.db")
        expected_db_path = os.path.join(test_file_path, "marsvision_expected.db")


        # If no expected db file is present, create one.
        # When we intentionally want to change database output,
        # we can delete the marsvision_expected file and run tests to create a new one.
        sw_test = SlidingWindow(model, "marsvision.db", 256, 256, 256, 256)
        if not os.path.isfile(expected_db_path):
            sw_test.db_path = expected_db_path
            sw_test.sliding_window_predict(image_list, metadata_list)
        sw_test.db_path = test_db_path
        sw_test.sliding_window_predict(image_list, metadata_list)

        test_conn = sqlite3.connect(test_db_path)
        expected_conn = sqlite3.connect(expected_db_path)
        

        # Asserting tables in both databases are the same
        test_global_table = pd.read_sql("SELECT * FROM global", test_conn)
        expected_global_table = pd.read_sql("SELECT * FROM global", expected_conn)
        
        test_image_table = pd.read_sql("SELECT * FROM windows", test_conn)
        expected_image_table = pd.read_sql("SELECT * FROM windows", expected_conn)

        test_metadata_table = pd.read_sql("SELECT * FROM metadata", test_conn)
        expected_metadata_table = pd.read_sql("SELECT * FROM metadata", expected_conn)

        #Remove output file; otherwise it gets appended to every time we run tests
        test_conn.close()
        expected_conn.close()
        os.remove(test_db_path)
        pd.testing.assert_frame_equal(test_metadata_table, expected_metadata_table)
        pd.testing.assert_frame_equal(test_image_table, expected_image_table)
        pd.testing.assert_frame_equal(test_global_table, expected_global_table)


