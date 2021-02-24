# Set up labels from text file.
# Might be useful for providing labels for pytorch.
import pandas as pd
lines = pd.read_csv(os.path.join(dataset_root, "labels-map-proj.txt"), 
                    delimiter=" ",  
                    header = None,
                   names=["file_name", "class_code"])
labels = dict(lines.values)