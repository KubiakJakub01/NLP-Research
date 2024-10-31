from torch import nn

from .hparams import VITSHparams
from .transformer import TextEncoder


class VITS(nn.Module):
    """VITS TTS model

    Paper: https://arxiv.org/pdf/2106.06103.pdf
    """

    def __init__(self, hparams: VITSHparams):
        super().__init__()
        self.hparams = hparams

        self.text_encoder = TextEncoder(
            n_vocab=hparams.num_chars,
            out_channels=hparams.hidden_channels,
            hidden_channels=hparams.hidden_channels,
            hidden_channels_ffn=hparams.hidden_channels_ffn_text_encoder,
            num_heads=hparams.num_heads_text_encoder,
            num_layers=hparams.num_layers_text_encoder,
            kernel_size=hparams.kernel_size_text_encoder,
            dropout_p=hparams.dropout_p_text_encoder,
            language_emb_dim=hparams.embedded_language_dim,
        )

    def forward(self, x):
        return x
