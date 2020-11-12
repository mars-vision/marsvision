import os
import numpy as np
from marsvision.utilities import DataLoader
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import LabelEncoder
import pickle

# Utility file to create testing files for use in our unit tests
current_dir = os.path.dirname(__file__)
test_image_path = os.path.join(current_dir, "test_images_loader")
test_model_path = os.path.join(current_dir, "testing_models")

# Create dataloader dataframe
dataloader = DataLoader(test_image_path, test_image_path)

# Create testing csv files
dataloader.run()

# Create file for sklearn logistic regression model
loaded_dataframe = dataloader.df
model_train_data = np.array(loaded_dataframe.iloc[:,0:5])
model_train_data_labels = loaded_dataframe["class_code"]

lr_model = LR()
lr_model.fit(model_train_data, loaded_dataframe["class_code"])

with open("test_lr_model.p", "wb") as out_file:
    pickle.dump(lr_model, out_file)