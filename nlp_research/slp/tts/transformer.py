import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .attend import RelativePositionMultiHeadAttention
from .modules import LayerNorm, LayerNorm2
from .utils import sequence_mask
from .wavenet import WaveNet


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int | None = None,
    ):
        """Text Encoder for VITS model.

        Args:
            n_vocab: Number of characters for the embedding layer.
            out_channels: Number of channels for the output.
            hidden_channels: Number of channels for the hidden layers.
            hidden_channels_ffn: Number of channels for the convolutional layers.
            num_heads: Number of attention heads for the Transformer layers.
            num_layers: Number of Transformer layers.
            kernel_size: Kernel size for the FFN layers in Transformer network.
            dropout_p: Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type='2',
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, lang_emb: torch.Tensor | None = None
    ):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
            - lang_emb: :math:`[L, H, 1]`
        """
        assert x.shape[0] == x_lengths.shape[0]
        B, T = x.shape
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat([x, rearrange(lang_emb, '(b l) (t h) 1 -> b t (l h)', b=B, T=T)], dim=-1)

        x = rearrange(x, 'b t h -> b h t')
        x_mask = rearrange(sequence_mask(x_lengths, T), 'b t -> b 1 t').to(x.dtype)  # [b, 1, t]

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


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

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask


class RelativePositionTransformer(nn.Module):
    """Transformer with Relative Potional Encoding.
    https://arxiv.org/abs/1803.02155

    Args:
        in_channels: number of channels of the input tensor.
        out_chanels: number of channels of the output tensor.
        hidden_channels: model hidden channels.
        hidden_channels_ffn: hidden channels of FeedForwardNetwork.
        num_heads: number of attention heads.
        num_layers: number of transformer layers.
        kernel_size: kernel size of feed-forward inner layers. Defaults to 1.
        dropout_p: dropout rate for self-attention and feed-forward inner layers_per_stack.
            Defaults to 0.
        rel_attn_window_size: relation attention window size.
            If 4, for each time step next and previous 4 time steps are attended.
            If default, relative encoding is disabled and it is a regular transformer.
            Defaults to None.
        input_length: input lenght to limit position encoding. Defaults to None.
        layer_norm_type: type "1" uses torch tensor operations and type "2" uses torch layer_norm
            primitive. Use type "2", type "1: is for backward compat. Defaults to "1".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int = 1,
        dropout_p: float = 0.0,
        rel_attn_window_size: int | None = None,
        input_length: int | None = None,
        layer_norm_type: str | None = '1',
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_channels_ffn = hidden_channels_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.rel_attn_window_size = rel_attn_window_size

        self.dropout = nn.Dropout(dropout_p)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for idx in range(self.num_layers):
            self.attn_layers.append(
                RelativePositionMultiHeadAttention(
                    hidden_channels if idx != 0 else in_channels,
                    hidden_channels,
                    num_heads,
                    rel_attn_window_size=rel_attn_window_size,
                    dropout_p=dropout_p,
                    input_length=input_length,
                )
            )
            if layer_norm_type == '1':
                self.norm_layers_1.append(LayerNorm(hidden_channels))
            elif layer_norm_type == '2':
                self.norm_layers_1.append(LayerNorm2(hidden_channels))
            else:
                raise ValueError(' [!] Unknown layer norm type')

            if hidden_channels != out_channels and (idx + 1) == self.num_layers:
                self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

            self.ffn_layers.append(
                FeedForwardNetwork(
                    hidden_channels,
                    hidden_channels if (idx + 1) != self.num_layers else out_channels,
                    hidden_channels_ffn,
                    kernel_size,
                    dropout_p=dropout_p,
                )
            )

            if layer_norm_type == '1':
                self.norm_layers_2.append(
                    LayerNorm(hidden_channels if (idx + 1) != self.num_layers else out_channels)
                )
            elif layer_norm_type == '2':
                self.norm_layers_2.append(
                    LayerNorm2(hidden_channels if (idx + 1) != self.num_layers else out_channels)
                )
            else:
                raise ValueError(' [!] Unknown layer norm type')

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        attn_mask = rearrange(x_mask, 'b 1 t -> b 1 1 t') * rearrange(x_mask, 'b 1 t -> b 1 t 1')
        for i in range(self.num_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.dropout(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)

            if (i + 1) == self.num_layers and hasattr(self, 'proj'):
                x = self.proj(x)

            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dropout_p: float = 0.0,
        causal: bool = False,
    ):
        """Feed Forward Inner layers for Transformer.

        Args:
            in_channels: input tensor channels.
            out_channels: output tensor channels.
            hidden_channels: inner layers hidden channels.
            kernel_size: conv1d filter kernel size.
            dropout_p: dropout rate. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size)
        self.conv_2 = nn.Conv1d(hidden_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = self.conv_1(self.padding(x * x_mask))
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, self._pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, self._pad_shape(padding))
        return x

    @staticmethod
    def _pad_shape(padding):
        pad_shape = [item for sublist in padding[::-1] for item in sublist]
        return pad_shape
