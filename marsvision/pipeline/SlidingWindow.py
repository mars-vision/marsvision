from marsvision.pipeline.Model import Model
from typing import TypeVar
import sqlite3
import pandas as pd
import cv2
from typing import List
import numpy as np
import yaml
import pdsc



class SlidingWindow:
    def __init__(self, model: Model, 
        db_path: str = "marsvision.db",
        window_length: int = 32,
        window_height: int = 32, 
        stride_x : int = 32, 
        stride_y: int = 32):
        """
            This class is responsible for running the sliding window pipeline,
            which will run through segments of an image with a window of user specified
            dimensions, and classify each one with a given machine learning model.   

            The results of the classification, as well as window and image information,
            is stored in a SQLite database.

            ------

            Parameters:

            db_path (str): File path of SQLite .db file.
            window_length (int): Length of window on the horizontal axis in pixels.
            window_height (int): Height of window on the vertical axis in pixels.
            stride_x (int): Stride of window along the horizontal axis in pixels.
            stride_y (int): Stride of window along the vertical axis in pixels.

        """
        self.window_length = window_length
        self.window_height = window_height
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.model = model
        self.db_path = db_path
        
    def sliding_window_predict(self, image_list: np.ndarray, metadata_list):
        """
            Runs the sliding window algorithm
            and makes a prediction for each window.
            
            Store the results in the SQLite database.

            -----

            Parameters

            image_list (numpy.ndarray): Image data represented as a numpy.ndarray.
            metadata_list: List of PDSC metadata objects.

        """

        # Open the DB connection and write global attributes
        self.conn = sqlite3.connect(self.db_path)
        self.create_sql_table()
        self.write_global_to_sql(metadata_list)

        # Number of images in the given batch
        batch_size = len(image_list)

        #  Get the primary key (auto incremented integer) of the new table we just wrote
        #  So that we can pass it onto the window
        c = self.conn.cursor()
        c.execute("SELECT id FROM global ORDER BY id DESC LIMIT " + str(batch_size))

        # Reverse returned id's to match up with the image_list and filename_list arrays.
        global_id_list = c.fetchall()
        global_id_list.reverse()
        global_id_list = list(zip(*global_id_list))[0]

        print(image_list)
        print(image_list.shape)
        image_width = image_list.shape[1]
        
        image_height = image_list.shape[2]

        for y in range(0, image_width, self.stride_x):
            for x in range(0, image_height, self.stride_y):
                # Slice window either to edge of image, or to end of window
                y_slice = min(image_width - y, self.window_height)
                x_slice = min(image_height - x, self.window_length)

                # Needs to be a vector of windows,
                # to send to the model predict function as a "list of images" 
                window_list = image_list[:, y:y_slice + y + 1, x:x_slice + x + 1, :]

                # Predict with model, store image coordinates of window in database
                self.write_window_to_sql(self.model.predict(window_list), metadata_list, x, y, global_id_list)

        # We're done with the database by this point, so close the connection 
        self.conn.close()

    def create_sql_table(self):
        """
            Helper that creates a global table if not present.
            
            This is used because we need to create entries with auto incremented IDs to pass to write_window_to_sql.

        """
        sql = """
            CREATE TABLE IF NOT EXISTS global (
                "id"	INTEGER,
                "observation_id"	TEXT,
                "stride_length_x"	INTEGER,
                "stride_length_y"	INTEGER,
                "window_length"	INTEGER,
                "window_height"	INTEGER,
                PRIMARY KEY("id" AUTOINCREMENT)
            );
        """
        c = self.conn.cursor()
        c.execute(sql)
                    
    def write_global_to_sql(self, metadata_list: List[str]):
        """
            Write entries for every image in the current batch of images.            

            The global table holds all data that is shared by all windows when we run the sliding window algorithm 
            over a particular image.
           
            This data includes the stride, window dimensions, and metadata associated with the particular image.

            -----

            Parameters
            
            metdata_list (List[str]): List of PDSC metadata objects. Can be used to derive the observation ID.
        """
        observation_ids = [metadata.observation_id for metadata in metadata_list ]
        row_count = len(metadata_list)
        image_dataframe = pd.DataFrame({
                    "stride_length_x": [self.stride_x] * row_count,
                    "stride_length_y": [self.stride_y] * row_count,
                    "window_length": [self.window_length] * row_count,
                    "window_height": [self.window_height] * row_count,
                    "observation_id": observation_ids
            }
        )
        image_dataframe.to_sql('global', con=self.conn, if_exists="append", index=False)

    def get_coordinates_from_metadata(self, metadata, pixel_coord_x, pixel_coord_y):
        rdr_localizer = pdsc.get_localizer(metadata)
        return rdr_localizer.pixel_to_latlon(pixel_coord_x, pixel_coord_y)
        

    def write_window_to_sql(self, prediction_list: List[int], metadata_list, window_coord_x: int, window_coord_y: int, global_id_list: np.ndarray):
        """
            Write a batch of inferences to the database. Include information about the window's location in its parent image,
            as well as a reference key to the parent image in the global table.

            ----

            Parameters
            prediction_list (np.ndarray): Batch of label inferences from the model.
            window_coord_x (int): x coordinate of the window on the parent image.
            window_coord_y (int): y coordinate of the window on the parent image.
            gloal_id (int): ID of parent image in Global table (which holds information about the image).
        """

        # Get window latitude/longitude coordinates here from PDSC queries.
        latlong_mid = [self.get_coordinates_from_metadata(metadata, window_coord_x + self.window_length / 2, window_coord_y + self.window_height / 2) for metadata in metadata_list]
        row_count = len(prediction_list)
        window_dataframe = pd.DataFrame({
                    "prediction": prediction_list,
                    "pixel_coord_x": [window_coord_x] * row_count,
                    "pixel_coord_y": [window_coord_y] *row_count,
                    "global_id": global_id_list,
                    "lat_x": latlong_mid[0],
                    "lat_y": latlong_mid[1]
                },
        )
        window_dataframe.to_sql('windows', con=self.conn, if_exists="append", index=False)
