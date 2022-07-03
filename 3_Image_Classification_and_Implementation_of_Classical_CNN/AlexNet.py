from collections import OrderedDict

import torch.nn as nn


# Conv1 - 卷积层 - 96 个 11*11 的 kernel
class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.Conv1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4))),
            ('ReLU', nn.ReLU(inplace=True)),
            ('LRN', nn.LocalResponseNorm(1))
        ]))

    def forward(self, x):
        return self.Conv1(x)


# Pool1 - 池化层 - 使用 3*3 核池化, 步长为 2
class Pool1(nn.Module):
    def __init__(self):
        super(Pool1, self).__init__()
        self.Pool1 = nn.Sequential(OrderedDict([
            ('Pool1', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, x):
        return self.Pool1(x)


# Conv2 - 卷积层 - 256 个 5*5 的 kernel, padding 为 2
class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.Conv2 = nn.Sequential(OrderedDict([
            ('Conv2', nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2)),
            ('ReLU', nn.ReLU(inplace=True)),
            ('LRN', nn.LocalResponseNorm(1))
        ]))

    def forward(self, x):
        return self.Conv2(x)


# Pool2 - 池化层 - 使用 3*3 核池化, 步长为 2
class Pool2(nn.Module):
    def __init__(self):
        super(Pool2, self).__init__()
        self.Pool2 = nn.Sequential(OrderedDict([
            ('Pool2', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, x):
        return self.Pool2(x)


# Conv3 - 卷积层 - 384 个 3*3 的 kernel, padding 为 1
class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.Conv3 = nn.Sequential(OrderedDict([
            ('Conv3', nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)),
            ('ReLU', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.Conv3(x)


# Conv4 - 卷积层 - 384 个 3*3 的 kernel, padding 为 1
class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        self.Conv4 = nn.Sequential(OrderedDict([
            ('Conv4', nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)),
            ('ReLU', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.Conv4(x)


# Conv5 - 卷积层 - 356 个 3*3 的 kernel, padding 为 1
class Conv5(nn.Module):
    def __init__(self):
        super(Conv5, self).__init__()
        self.Conv5 = nn.Sequential(OrderedDict([
            ('Conv5', nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)),
            ('ReLU', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.Conv5(x)


# Pool5 - 池化层 - 使用 3*3 核池化, 步长为 2
class Pool5(nn.Module):
    def __init__(self):
        super(Pool5, self).__init__()
        self.Pool5 = nn.Sequential(OrderedDict([
            ('Pool5', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, x):
        return self.Pool5(x)


# FC6 - 全连接层
class FC6(nn.Module):
    def __init__(self):
        super(FC6, self).__init__()

        self.FC6 = nn.Sequential(OrderedDict([
            ('FC6', nn.Linear(1024, 4096)),
            ('ReLU', nn.ReLU(inplace=True)),
            ('Dropout', nn.Dropout(p=0.1))
        ]))

    def forward(self, x):
        return self.FC6(x)


# FC7 - 全连接层
class FC7(nn.Module):
    def __init__(self):
        super(FC7, self).__init__()

        self.FC7 = nn.Sequential(OrderedDict([
            ('FC7', nn.Linear(4096, 4096)),
            ('ReLU', nn.ReLU(inplace=True)),
            ('Dropout', nn.Dropout(p=0.1))
        ]))

    def forward(self, x):
        return self.FC7(x)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            Conv1(),
            Pool1(),
            Conv2(),
            Pool2(),
            Conv3(),
            Conv4(),
            Conv5(),
            Pool5()
        )

        self.classifier = nn.Sequential(
            FC6(),
            FC7()
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.view(x.size(0), -1)
        output = self.classifier(feature)
        return output
