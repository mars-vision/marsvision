from marsvision.pipeline import Model
import sqlite3
import pandas as pd

class SlidingWindow:
    def __init__(self, model: Model, 
        window_length: int = 32,
        window_height: int = 32, 
        stride_x : int = 32, 
        stride_y: int = 32):
        self.window_length = window_length
        self.window_height = window_height
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.model = model
        self.conn = sqlite3.connect("marsvision.db")

    def sliding_window_predict(self, image):
        self.write_image_to_sql()
        for y in range(0, image.shape[0], self.stride_x):
            for x in range(0, image.shape[1], self.stride_y):
                # Store window attributes in a dictionary structure
                current_window_dict = {}
                # Slice window either to edge of image, or to end of window
                y_slice = min(image.shape[0] - y, self.window_height)
                x_slice = min(image.shape[1] - x, self.window_length)
                window = image[y:y_slice + y + 1, x:x_slice + x + 1]
                
                # Predict with model, store image coordinates of window in database
                self.write_window_to_sql(self.model.predict(window), x, y)
            
    def write_image_to_sql(self):
        image_data_row = {
            "stride_length_x": self.stride_x,
            "stride_length_y": self.stride_y,
            "window_length": self.window_length,
            "window_height":  self.window_height
        }
        image_dataframe = pd.DataFrame(data=image_data_row, index=[0])
        image_dataframe.to_sql('images', con=self.conn, if_exists="append", index=False)

    def write_window_to_sql(self, prediction, x, y):
        # TODO: image_id should match the corresponding image primary key
        window_row = {
            "prediction": prediction, 
            "coord_x": x,
            "coord_y": y,
            "image_id": 0 
        }
        window_row_data = pd.DataFrame(data=window_row, index=[0])
        window_row_data.to_sql('windows', con=self.conn, if_exists="append", index=False)
        

