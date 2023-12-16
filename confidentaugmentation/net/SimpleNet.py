import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        self.norm1 = torch.nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="relu")
        self.norm2 = torch.nn.BatchNorm2d(3)

        self.conv3 = nn.Conv2d(3, 16, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv3.weight, mode="fan_in", nonlinearity="relu")
        self.norm3 = torch.nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv4.weight, mode="fan_in", nonlinearity="relu")
        self.norm4 = torch.nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 24, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv5.weight, mode="fan_in", nonlinearity="relu")
        self.norm5 = torch.nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 24, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv6.weight, mode="fan_in", nonlinearity="relu")
        self.norm6 = torch.nn.BatchNorm2d(24)

        self.conv7 = nn.Conv2d(24, 32, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv7.weight, mode="fan_in", nonlinearity="relu")
        self.norm7 = torch.nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv8.weight, mode="fan_in", nonlinearity="relu")
        self.norm8 = torch.nn.BatchNorm2d(32)

        self.conv9 = nn.Conv2d(32, 48, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv9.weight, mode="fan_in", nonlinearity="relu")
        self.norm9 = torch.nn.BatchNorm2d(48)
        self.conv10 = nn.Conv2d(48, 48, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv10.weight, mode="fan_in", nonlinearity="relu")
        self.norm10 = torch.nn.BatchNorm2d(48)

        self.fc1 = nn.Linear(48, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

    def forward(self, x):
        dropout_rate = 0.

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.norm6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv7(x)
        x = self.norm7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.norm8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv9(x)
        x = self.norm9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = self.norm10(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x