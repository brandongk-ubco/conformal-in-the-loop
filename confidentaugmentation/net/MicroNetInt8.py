import torch
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.nn import Linear8bitLt

class MicroNetInt8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = Linear8bitLt(784, num_classes, has_fp16_weights=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x
