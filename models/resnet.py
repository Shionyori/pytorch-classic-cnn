import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1conv=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):

        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(x + y) # 跳转连接
        return y
    
class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            ResidualBlock(64, 64, use_1conv=False, stride=1),
            ResidualBlock(64, 64, use_1conv=False, stride=1)
        )
        self.b3 = nn.Sequential(
            ResidualBlock(64, 128, use_1conv=True, stride=2),
            ResidualBlock(128, 128, use_1conv=False, stride=1)
        )
        self.b4 = nn.Sequential(
            ResidualBlock(128, 256, use_1conv=True, stride=2),
            ResidualBlock(256, 256, use_1conv=False, stride=1)
        )
        self.b5 = nn.Sequential(
            ResidualBlock(256, 512, use_1conv=True, stride=2),
            ResidualBlock(512, 512, use_1conv=False, stride=1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x
