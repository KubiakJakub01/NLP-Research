from torch import nn


class VITS(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, x):
        return x
