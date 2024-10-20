from torch import nn

from .hparams import VITSHparams


class VITS(nn.Module):
    def __init__(self, hparams: VITSHparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, x):
        return x
