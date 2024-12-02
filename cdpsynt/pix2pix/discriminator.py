from torch import nn
from collections import OrderedDict
import torch.nn.functional as F


__all__ = ["Discriminator"]


class Discriminator(nn.Module):
    def __init__(self, channels_in: int, dropout_p: float = 0.4):
        super(Discriminator, self).__init__()
        self.channels_in: int = channels_in
        self.dropout_p: float = dropout_p
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(self.channels_in, 128, 3, stride=2, padding=4, padding_mode='reflect')),
                    ("bn1", nn.BatchNorm2d(128)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(128, 256, 3, stride=2, padding=4, padding_mode='reflect')),
                    ("bn2", nn.BatchNorm2d(256)),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(256, 512, 3)),
                    ("dropout3", nn.Dropout2d(p=self.dropout_p)),
                    ("bn3", nn.BatchNorm2d(512)),
                    ("relu3", nn.ReLU()),
                    ("conv4", nn.Conv2d(512, 1024, 3)),
                    ("dropout4", nn.Dropout2d(p=self.dropout_p)),
                    ("bn4", nn.BatchNorm2d(1024)),
                    ("relu4", nn.ReLU()),
                    ("conv5", nn.Conv2d(1024, 512, 3, stride=2, padding=4, padding_mode='reflect')),
                    ("dropout5", nn.Dropout2d(p=self.dropout_p)),
                    ("bn5", nn.BatchNorm2d(512)),
                    ("relu5", nn.ReLU()),
                    ("conv6", nn.Conv2d(512, 256, 3, stride=2, padding=4, padding_mode='reflect')),
                    ("dropout6", nn.Dropout2d(p=self.dropout_p)),
                    ("bn6", nn.BatchNorm2d(256)),
                    ("relu6", nn.ReLU()),
                    ("conv7", nn.Conv2d(256, 128, 3)),
                    ("bn7", nn.BatchNorm2d(128)),
                    ("relu7", nn.ReLU()),
                    ("conv8", nn.Conv2d(128, 1, 3)),
                ]
            )
        )

    def forward(self, x):
        x = self.model(x)
        x = F.sigmoid(x)
        return x
