import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch import nn

__all__ = ["Generator"]


class Generator(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dropout_p=0.4):
        super(Generator, self).__init__()
        self.dropout_p = dropout_p
        filters = [1024, 512, 256, 128, 16]
        self.unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=channels_in,
            classes=channels_out,
            activation=None,
            encoder_depth=len(filters),
            decoder_channels=filters,
        )

        for idx in range(1, 3):
            self.unet.decoder.blocks[idx].conv1.add_module(
                "3", nn.Dropout2d(p=self.dropout_p)
            )

        for module in self.unet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def forward(self, x):
        x = self.unet(x)
        x = F.relu(x)
        return x
