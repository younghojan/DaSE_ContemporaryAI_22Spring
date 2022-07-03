from collections import OrderedDict

import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()

        self.ConvLayer = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)),
            ('BN', nn.BatchNorm2d(out_channels)),
            ('ReLU', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.ConvLayer(x)


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCLayer, self).__init__()

        self.FCLayer = nn.Sequential(OrderedDict([
            ('FC', nn.Linear(in_features, out_features)),
            ('Dropout', nn.Dropout())
        ]))

    def forward(self, x):
        return self.FCLayer(x)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            ConvLayer(in_channels=1, out_channels=64),
            ConvLayer(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            ConvLayer(in_channels=64, out_channels=128),
            ConvLayer(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            ConvLayer(in_channels=128, out_channels=256),
            ConvLayer(in_channels=256, out_channels=256),
            ConvLayer(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            ConvLayer(in_channels=256, out_channels=512),
            ConvLayer(in_channels=512, out_channels=512),
            ConvLayer(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            ConvLayer(in_channels=512, out_channels=512),
            ConvLayer(in_channels=512, out_channels=512),
            ConvLayer(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.classifier = nn.Sequential(
            FCLayer(512, 4096),
            FCLayer(4096, 4096),
            FCLayer(4096, 10)
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.view(x.size(0), -1)
        output = self.classifier(feature)
        return output
