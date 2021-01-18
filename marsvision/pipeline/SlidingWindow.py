from marsvision.pipeline.Model import Model
from typing import TypeVar
import sqlite3
import pandas as pd
import cv2

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
        
    def sliding_window_predict(self, image, filename: str = None):
        """
            Runs the sliding window algorithm
            and makes a prediction for each window.
            
            Store the results in the SQLite database.

            -----

            Parameters

            image: Image data represented as a numpy.ndarray.

        """

        # Open the DB connection and write global attributes
        self.conn = sqlite3.connect(self.db_path)
        self.write_global_to_sql(str(filename))

        #  Get the primary key (auto incremented integer) of the new table we just wrote
        #  So that we can pass it onto the window
        c = self.conn.cursor()
        c.execute("SELECT id FROM global ORDER BY id DESC LIMIT 1")
        global_id = c.fetchone()[0]

        for y in range(0, image.shape[0], self.stride_x):
            for x in range(0, image.shape[1], self.stride_y):

                # Slice window either to edge of image, or to end of window
                y_slice = min(image.shape[0] - y, self.window_height)
                x_slice = min(image.shape[1] - x, self.window_length)
                window = image[y:y_slice + y + 1, x:x_slice + x + 1]
                
                # Predict with model, store image coordinates of window in database
                self.write_window_to_sql(self.model.predict(window), x, y, global_id)

        # We're done with the database by this point, so close the connection 
        self.conn.close()
            
    def write_global_to_sql(self, filename: str):
        """
            Write to the global table in the database.

            The global table holds all data that is shared by all windows when we run the sliding window algorithm 
            over a particular image.
           
            This data includes the stride, window dimensions, and metadata associated with the particular image.

            -----

            Parameters
            
            filename(str): File name of the image. Can be used to derive the observation ID.


        """
        image_data_row = {
            "stride_length_x": self.stride_x,
            "stride_length_y": self.stride_y,
            "window_length": self.window_length,
            "window_height":  self.window_height,
            "filename": filename
        }
        image_dataframe = pd.DataFrame(data=image_data_row)
        image_dataframe.to_sql('global', con=self.conn, if_exists="append", index=False)
        
      

    def write_window_to_sql(self, prediction: int, window_coord_x: int, window_coord_y: int, global_id: int):
        window_row = {
            "prediction": prediction, 
            "coord_x": window_coord_x,
            "coord_y": window_coord_y,
            "global_id": global_id
        }
        window_row_data = pd.DataFrame(data=window_row, index=[0])
        window_row_data.to_sql('windows', con=self.conn, if_exists="append", index=False)
