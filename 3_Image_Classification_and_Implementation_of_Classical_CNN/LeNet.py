from collections import OrderedDict

import torch.nn as nn


# C1 层 - 卷积层 - 6 个 5*5 的 kernel
class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        self.C1 = nn.Sequential(OrderedDict([
            ('C1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('Sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.C1(x)


# S2 层 - 池化层 - 使用 2*2 核池化
class S2(nn.Module):
    def __init__(self):
        super(S2, self).__init__()
        self.S2 = nn.Sequential(OrderedDict([
            ('S2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('Dropout', nn.Dropout(p=0.1))
        ]))

    def forward(self, x):
        return self.S2(x)


# C3 层 - 卷积层 - 16 个 5x5 的 kernel
class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        self.C3 = nn.Sequential(OrderedDict([
            ('C3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('Sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.C3(x)


# S4 层 - 池化层 - 使用 2*2 核池化
class S4(nn.Module):
    def __init__(self):
        super(S4, self).__init__()
        self.S4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        return self.S4(x)


# C5 层 - 卷积层 - 120 个 5x5 的 kernel
class C5(nn.Module):
    def __init__(self):
        super(C5, self).__init__()

        self.C5 = nn.Sequential(OrderedDict([
            ('C5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('Sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.C5(x)


# F6 层 - 全连接层 - 84 个单元
class F6(nn.Module):
    def __init__(self):
        super(F6, self).__init__()

        self.F6 = nn.Sequential(OrderedDict([
            ('F6', nn.Linear(120, 84)),
            ('Sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.F6(x)


class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()

        self.Output = nn.Sequential(OrderedDict([
            ('F', nn.Linear(84, 10)),
            # ('Softmax', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        return self.Output(x)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            C1(),
            S2(),
            C3(),
            S4(),
            C5()
        )
        self.classifier = nn.Sequential(
            F6(),
            Output()
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.view(x.size(0), -1)
        output = self.classifier(feature)
        return output
