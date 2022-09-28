import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):  # pragma: no cover
    def __init__(self):
        super(ConvNet, self).__init__()
        # Basic CNN:
        # Convolution -> Relu -> Linear transformation -> Relu ->  Output (10 features)
        self.conv = nn.Conv2d(1, 1, 3)
        self.linear = nn.Linear(26 * 26, 10)

    def forward(self, x):
        # Define the forward pass here
        x = F.relu(self.conv(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear(x))
        return x

    def num_flat_features(self, x):
        # Image dimensionality
        # All dimensions except batch dimension
        # Batch dimension being # of inputs
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
