import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 3, 3, padding=1, padding_mode="zeros")
        # nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        # self.conv2 = nn.Conv2d(3, 1, 3, padding=1, padding_mode="zeros")
        # nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        self.fc1 = nn.Linear(784, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

    def forward(self, x):

        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x