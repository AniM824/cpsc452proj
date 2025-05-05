import torch
import numpy as np
from torch import nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, AvgPool2d, Softmax, BatchNorm2d, Dropout
from torch.optim import SGD

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Code from: https://github.com/Rob-Christian/MiniGoogleNet?tab=readme-ov-file
## Inception Model Creation

# For convolutional module
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, filt_size, strd, pad):
        super(ConvBlock, self).__init__()
        self.conv2d = Conv2d(in_channels=cin, out_channels=cout, kernel_size=filt_size, stride=strd, padding=pad)
        self.batch_norm = BatchNorm2d(num_features=cout)
        self.relu = ReLU()
    def forward(self, x):
        return self.relu(self.batch_norm(self.conv2d(x)))

# For inception module
class InceptionBlock(nn.Module):
    def __init__(self, cin, cout1, cout3):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(cin, cout1, filt_size=1, strd=1, pad=0)
        self.branch2 = ConvBlock(cin, cout3, filt_size=3, strd=1, pad=1)
    def forward(self, x):
        branches = (self.branch1, self.branch2)
        return torch.cat([branch(x) for branch in branches], 1)

# For downsample module
class DownsampleBlock(nn.Module):
    def __init__(self, cin, cout3):
        super(DownsampleBlock, self).__init__()
        self.branch1 = ConvBlock(cin, cout3, filt_size=3, strd=2, pad=0)
        self.branch2 = MaxPool2d(kernel_size=3, stride=2)
    def forward(self, x):
        branches = (self.branch1, self.branch2)
        return torch.cat([branch(x) for branch in branches], 1)

# For the whole small model
class baseline_small_inception(nn.Module):
    def __init__(self, num_classes=10):
        super(baseline_small_inception, self).__init__()
        self.conv1 = ConvBlock(cin=3, cout=96, filt_size=3, strd=1, pad=0)
        self.inception1a = InceptionBlock(cin=96, cout1=32, cout3=32)
        self.inception1b = InceptionBlock(cin=64, cout1=32, cout3=48)
        self.downsample1 = DownsampleBlock(cin=80, cout3=80)
        self.inception2a = InceptionBlock(cin=160, cout1=112, cout3=48)
        self.inception2b = InceptionBlock(cin=160, cout1=96, cout3=64)
        self.inception2c = InceptionBlock(cin=160, cout1=80, cout3=80)
        self.inception2d = InceptionBlock(cin=160, cout1=48, cout3=96)
        self.downsample2 = DownsampleBlock(cin=144, cout3=96)
        self.inception3a = InceptionBlock(cin=240, cout1=176, cout3=160)
        self.inception3b = InceptionBlock(cin=336, cout1=176, cout3=160)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = Linear(336, num_classes)
        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.downsample1(x)
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.inception2c(x)
        x = self.inception2d(x)
        x = self.downsample2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fully_connected(x)
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    model = baseline_small_inception(num_classes=10)
    print(model)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of trainable parameters: {total_params:,}")