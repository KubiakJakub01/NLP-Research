from torch import nn

from .hparams import VITSHparams
from .networks import PosteriorEncoder
from .transformer import TextEncoder


class VITS(nn.Module):
    """VITS TTS model

    Paper: https://arxiv.org/pdf/2106.06103.pdf
    """

    def __init__(self, hparams: VITSHparams):
        super().__init__()
        self.hparams = hparams

        self.text_encoder = TextEncoder(
            n_vocab=self.hparams.num_chars,
            out_channels=self.hparams.hidden_channels,
            hidden_channels=self.hparams.hidden_channels,
            hidden_channels_ffn=self.hparams.hidden_channels_ffn_text_encoder,
            num_heads=self.hparams.num_heads_text_encoder,
            num_layers=self.hparams.num_layers_text_encoder,
            kernel_size=self.hparams.kernel_size_text_encoder,
            dropout_p=self.hparams.dropout_p_text_encoder,
            language_emb_dim=self.hparams.embedded_language_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            in_channels=self.hparams.out_channels,
            out_channels=self.hparams.hidden_channels,
            hidden_channels=self.hparams.hidden_channels,
            kernel_size=self.hparams.kernel_size_posterior_encoder,
            dilation_rate=self.hparams.dilation_rate_posterior_encoder,
            num_layers=self.hparams.num_layers_posterior_encoder,
            cond_channels=self.hparams.embedded_language_dim,
        )

    def forward(self, x):
        return x
