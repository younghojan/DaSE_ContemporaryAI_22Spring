import torch.nn as nn


class VisualClf(nn.Module):
    def __init__(self, img_hidden_size, num_classes):
        super(VisualClf, self).__init__()
        self.fc = nn.Linear(img_hidden_size, num_classes)

    def forward(self, x):
        return self.fc(x)
