import torch
from einops import rearrange
from torch import nn

from .modules import LayerNorm
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


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, 'channels should be divisible by 2'
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WaveNet(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.

        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if reverse:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

        x1 = m + x1 * torch.exp(log_scale) * x_mask
        x = torch.cat([x0, x1], 1)
        logdet = torch.sum(log_scale, [1, 2])
        return x, logdet


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows: int = 4,
        cond_channels: int = 0,
    ):
        """Redisual Coupling blocks for VITS flow layers.

        Args:
            channels: Number of input and output tensor channels.
            hidden_channels: Number of hidden network channels.
            kernel_size: Kernel size of the WaveNet layers.
            dilation_rate: Dilation rate of the WaveNet layers.
            num_layers: Number of the WaveNet layers.
            num_flows: Number of Residual Coupling blocks. Defaults to 4.
            cond_channels: Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.

        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class DurationPredictor(nn.Module):
    """Glow-TTS duration prediction model.

    ::

        [2 x (conv1d_kxk -> relu -> layer_norm -> dropout)] -> conv1d_1x1 -> durs

    Args:
        in_channels (int): Number of channels of the input tensor.
        hidden_channels (int): Number of hidden channels of the network.
        kernel_size (int): Kernel size for the conv layers.
        dropout_p (float): Dropout rate used after each conv layer.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dropout_p,
        cond_channels=None,
        language_emb_dim=None,
    ):
        super().__init__()

        # add language embedding dim in the input
        if language_emb_dim:
            in_channels += language_emb_dim

        # class arguments
        self.in_channels = in_channels
        self.filter_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        # layers
        self.drop = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(hidden_channels)
        self.conv_2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(hidden_channels)
        # output layer
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        if cond_channels is not None and cond_channels != 0:
            self.cond = nn.Conv1d(cond_channels, in_channels, 1)

        if language_emb_dim != 0 and language_emb_dim is not None:
            self.cond_lang = nn.Conv1d(language_emb_dim, in_channels, 1)

    def forward(self, x, x_mask, g=None, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if g is not None:
            x = x + self.cond(g)

        if lang_emb is not None:
            x = x + self.cond_lang(lang_emb)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
