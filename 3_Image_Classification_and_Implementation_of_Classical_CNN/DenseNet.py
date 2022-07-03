import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()

        self.ConvLayer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.ConvLayer(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, bottle_neck_size, growth_rate):
        super(DenseLayer, self).__init__()

        self.DenseLayer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottle_neck_size, kernel_size=(1, 1)),

            nn.BatchNorm2d(bottle_neck_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottle_neck_size, growth_rate, kernel_size=(3, 3), padding=1)
        )

    def forward(self, *prev_feature_maps):
        return self.DenseLayer(torch.cat(prev_feature_maps, dim=1))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, layer_counts, growth_rate):
        super(DenseBlock, self).__init__()
        self.Layers = []
        for i in range(layer_counts):
            curr_input_channel = in_channels + i * growth_rate
            bottleneck_size = 4 * growth_rate
            layer = DenseLayer(curr_input_channel, bottleneck_size, growth_rate)
            self.Layers.append(layer.cuda())

    def forward(self, init_features):
        features = [init_features]
        for layer in self.Layers:
            layer_out = layer(*features)
            features.append(layer_out)

        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.TransitionLayer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.TransitionLayer(x)


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            ConvLayer(),

            DenseBlock(in_channels=64, layer_counts=6, growth_rate=32),
            TransitionLayer(in_channels=256, out_channels=128),

            DenseBlock(in_channels=128, layer_counts=12, growth_rate=32),
            TransitionLayer(in_channels=512, out_channels=256),

            DenseBlock(in_channels=256, layer_counts=24, growth_rate=32),
            TransitionLayer(in_channels=1024, out_channels=512),

            DenseBlock(in_channels=512, layer_counts=16, growth_rate=32)
        )

        self.Pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Linear(1024, 10)

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = self.Pool(feature)
        feature = feature.view(x.size(0), -1)
        output = self.classifier(feature)
        return output
