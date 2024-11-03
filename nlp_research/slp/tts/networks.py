import torch
from einops import rearrange
from torch import nn

from .utils import sequence_mask
from .wavenet import WaveNet


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels: int = 0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) ->
            conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels: Number of input tensor channels.
            out_channels: Number of output tensor channels.
            hidden_channels: Number of hidden channels.
            kernel_size: Kernel size of the WaveNet convolution layers.
            dilation_rate: Dilation rate of the WaveNet layers.
            num_layers: Number of the WaveNet layers.
            cond_channels: Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers,
            c_in_channels=cond_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor | None = None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        _, _, T = x.shape
        x_mask = rearrange(sequence_mask(x_lengths, T), 'b t -> b 1 t').to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask
