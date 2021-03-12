import torch
import yaml
from marsvision.config_path import CONFIG_PATH
import torch.nn as nn

with open(CONFIG_PATH) as yaml_cfg:
    config = yaml.load(yaml_cfg)

def alexnet():
    num_classes = config["pytorch_cnn_parameters"]["num_output_classes"]
    alexnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    alexnet_model.classifier[6] = nn.Linear(4096, num_classes)
    return alexnet_model