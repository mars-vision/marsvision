import os
from marsvision.utilities import DataLoader

current_dir = os.path.dirname(__file__)
test_image_path = os.path.join(current_dir, "test_images_loader")
loader_with_folder_names = DataLoader(test_image_path, test_image_path)
loader_with_folder_names.run()
