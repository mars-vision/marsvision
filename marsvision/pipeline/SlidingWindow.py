from marsvision.pipeline import Model

def sliding_window_predict(image, 
    model: Model, 
    window_length: int = 32,
    window_height: int = 32, 
    stride_x : int = 32, 
    stride_y: int = 32):
    
    window_data = []
    for y in range(0, image.shape[0], stride_x):
        for x in range(0, image.shape[1], stride_y):
            # Store window attributes in a dictionary structure
            current_window_dict = {}
            # Slice window either to edge of image, or to end of window
            y_slice = min(image.shape[0] - y, window_height)
            x_slice = min(image.shape[1] - x, window_length)
            window = image[y:y_slice + y + 1, x:x_slice + x + 1]
            
            # Predict with model, store image coordinates of window
            current_window_dict["prediction"] = model.predict(window)
            current_window_dict["coordinates"] = (y, y_slice + y, x, x_slice + x)
            window_data.append(current_window_dict)

    return window_data