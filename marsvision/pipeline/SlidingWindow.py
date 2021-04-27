from marsvision.pipeline.Model import Model
from typing import TypeVar
import sqlite3
import pandas as pd
import cv2
from typing import List
import numpy as np
import yaml
import pdsc
from pdsc.metadata import json_dumps

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

        # Write metadata to a SQL table.
        # Use global id's to tie metadata to parent images.
        self.write_metadata_to_sql(metadata_list)

        # Get the width of the widest image, 
        # and height of the tallest image.
        image_widths = [image.shape[1] for image in image_list]
        image_heights = [image.shape[0] for image in image_list]
        max_width = max(image_widths)
        max_height = max(image_heights)

        # Handle images of different sizes by 
        # sliding the window over the dimensions of the largest image,
        # and checking if the same window is valid for smaller images.
        for y in range(0, max_width, self.stride_x):
            for x in range(0, max_height, self.stride_y):
                window_list = []
                metadata_filtered_list = []
                global_id_filtered_list = []

                # Only include windows from images
                # if the indexing is not out of range.
                for i in range(len(image_list)):
                    if x < image_widths[i] and y < image_heights[i]:
                        # Slice window either to edge of this image, or to end of window
                        y_slice = min(image_heights[i] - y, self.window_height)
                        x_slice = min(image_widths[i] - x, self.window_length)
                        window_list.append(image_list[i][y:y_slice + y + 1, x:x_slice + x + 1, :])
                        metadata_filtered_list.append(metadata_list[i])
                        global_id_filtered_list.append(global_id_list[i])
                        
                # Don't do anything if there is no input
                # This occurs if the sliding window does not land on any images.
                # i.e. when the window overflows on both dimensions.
                if len(window_list) == 0:
                    continue
                # Predict with model, store image coordinates of window in database

                # Write the window to a SQL table.
                # Pass the filtered list of metadata and global id's
                self.write_window_to_sql(self.model.predict(window_list), metadata_filtered_list, x, y, global_id_filtered_list)

                

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
                "product_id"    TEXT,
                "stride_length_x"	INTEGER,
                "stride_length_y"	INTEGER,
                "window_length"	INTEGER,
                "window_height"	INTEGER,
                "model_type" TEXT,
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
            
            metdata_list (List[str]): List of PDSC metadata objects. Can be used to derive the observation ID and product ID.
        """

        # Observation IDs are not unique as observations product multiple files,
        #  though they tell us which observation goes with which files.
        observation_ids = [metadata.observation_id for metadata in metadata_list]

        # Product ID is the unique identifier for a given image.
        # This is the foreign key to the metadata table.
        # This is how we can tie entries in this table to image metadata.
        product_ids = [metadata.product_id for metadata in metadata_list]
        row_count = len(metadata_list)
        image_dataframe = pd.DataFrame({
                    "stride_length_x": [self.stride_x] * row_count,
                    "stride_length_y": [self.stride_y] * row_count,
                    "window_length": [self.window_length] * row_count,
                    "window_height": [self.window_height] * row_count,
                    "model_type": [self.model.model_type] * row_count,
                    "product_id": product_ids,
                    "observation_id": observation_ids
            }
        )
        image_dataframe.to_sql('global', con=self.conn, if_exists="append", index=False)

    def get_coordinates_from_metadata(self, metadata, pixel_coord_x, pixel_coord_y):
        rdr_localizer = pdsc.get_localizer(metadata)
        latlong = rdr_localizer.pixel_to_latlon(pixel_coord_y, pixel_coord_x)
        return latlong

    def write_metadata_to_sql(self, metadata_list):
        """
            Write image metadata to a separate table.

            This is the metadata associated with the images via the PDSC api.
        """
        # json_dumps is part of the pdsc API. Parses a PDSC metadata object to JSON format.
        metadata_dataframe = pd.read_json(json_dumps(metadata_list))
        
        # Dates are formatted as weird JSON objects.
        # These lines extract the formatted string.
        # E.g. 2007-09-23T00:22:40.000000'
        metadata_dataframe["start_time"] = metadata_dataframe["start_time"].apply(lambda datedict: datedict["__datetime__"]["__val__"])
        metadata_dataframe["observation_start_time"] = metadata_dataframe["observation_start_time"].apply(lambda datedict: datedict["__datetime__"]["__val__"])
        metadata_dataframe["stop_time"] = metadata_dataframe["stop_time"].apply(lambda datedict: datedict["__datetime__"]["__val__"])

        # Typecasting everything to str avoids errors in the to_sql call.
        metadata_dataframe = metadata_dataframe.applymap(str)

        # Write to the metadata table.
        metadata_dataframe.to_sql('metadata', con=self.conn, if_exists="append", index=False)


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

        # Get window latitude/longitude information for each window.
        # Calculate the min and max latitudes and longitudes of our window.
        # Done by getting them for the top left and bottom right corners (i.e. (min, min) and (max, max))
        latlong_min = []
        latlong_max = []
        for metadata in metadata_list:
            # Put together lists of tuples: (latitude, longitude)
            latlong_min.append(self.get_coordinates_from_metadata(metadata, window_coord_x, window_coord_y))
            latlong_max.append(self.get_coordinates_from_metadata(metadata, window_coord_x + self.window_length, window_coord_y + self.window_height))

        # Rearrange the list of coordinate pairs so that we have a tuple of latitudes,
        # and a tuple of longitudes.
        latlong_min = list(zip(*latlong_min))
        latlong_max = list(zip(*latlong_max))


        # Put together parallel lists to pass to the dataframe
        min_latitudes = latlong_max[0]
        min_longitudes = latlong_min[1]
        max_latitudes = latlong_min[0]
        max_longitudes = latlong_max[1]
        
        row_count = len(prediction_list)
        window_dataframe = pd.DataFrame({
                    "prediction": prediction_list,
                    "pixel_coord_x": [window_coord_x] * row_count,
                    "pixel_coord_y": [window_coord_y] *row_count,
                    "global_id": global_id_list,
                    "minimum_longitude": min_longitudes,
                    "minimum_latitude": min_latitudes,
                    "maximum_longitude": max_longitudes,
                    "maximum_latitude": max_latitudes
                },
        )
        window_dataframe.to_sql('windows', con=self.conn, if_exists="append", index=False)
