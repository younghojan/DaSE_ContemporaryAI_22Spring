from collections import OrderedDict

import torch.nn as nn
from torch.nn import functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample):
        super(ResNetBlock, self).__init__()
        self.down_sample = down_sample

        self.DownSample = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2))),
            ('BN', nn.BatchNorm2d(out_channels))
        ]))

        if down_sample:
            self.Conv1 = nn.Sequential(OrderedDict([
                ('Conv1', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1)),
                ('BN1', nn.BatchNorm2d(out_channels)),
                ('ReLU', nn.ReLU(inplace=True))
            ]))
            self.Conv2 = nn.Sequential(OrderedDict([
                ('Conv2', nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)),
                ('BN2', nn.BatchNorm2d(out_channels))
            ]))
        else:
            self.Conv1 = nn.Sequential(OrderedDict([
                ('Conv1', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)),
                ('BN1', nn.BatchNorm2d(out_channels)),
                ('ReLU', nn.ReLU(inplace=True))
            ]))
            self.Conv2 = nn.Sequential(OrderedDict([
                ('Conv2', nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)),
                ('BN2', nn.BatchNorm2d(out_channels))
            ]))

    def forward(self, x):
        output = self.Conv1(x)
        output = self.Conv2(output)

        if self.down_sample:
            return F.relu(self.DownSample(x) + output)
        else:
            return F.relu(x + output)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.Conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.BN = nn.BatchNorm2d(64)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Layer1 = nn.Sequential(
            ResNetBlock(in_channels=64, out_channels=64, down_sample=False),
            ResNetBlock(in_channels=64, out_channels=64, down_sample=False)
        )

        self.Layer2 = nn.Sequential(
            ResNetBlock(in_channels=64, out_channels=128, down_sample=True),
            ResNetBlock(in_channels=128, out_channels=128, down_sample=False)
        )

        self.Layer3 = nn.Sequential(
            ResNetBlock(in_channels=128, out_channels=256, down_sample=True),
            ResNetBlock(in_channels=256, out_channels=256, down_sample=False)
        )

        self.Layer4 = nn.Sequential(
            ResNetBlock(in_channels=256, out_channels=512, down_sample=True),
            ResNetBlock(in_channels=512, out_channels=512, down_sample=False)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.FC = nn.Linear(512, 10)

    def forward(self, x):
        out = self.Conv(x)
        out = self.BN(out)
        out = self.MaxPool(out)
        out = self.Layer1(out)
        out = self.Layer2(out)
        out = self.Layer3(out)
        out = self.Layer4(out)
        out = self.AvgPool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.FC(out)
        return out
