import torch
import yaml
from marsvision.path_definitions import CONFIG_PATH
import torch.nn as nn

with open(CONFIG_PATH) as yaml_cfg:
    config = yaml.load(yaml_cfg)

def alexnet_grayscale():
    num_classes = config["pytorch_cnn_parameters"]["num_output_classes"]
    alexnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    # Modify some of the layers to support a grayscale image.
    # Also bring down the number of connections in the classifier to support fewer output classes.
    alexnet_model.features[0] = nn.Conv2d(1, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2))
    alexnet_model.classifier[4] = nn.Linear(4096, 1024)
    alexnet_model.classifier[6] = nn.Linear(1024, num_classes)
    return alexnet_model

