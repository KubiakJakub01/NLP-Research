from torch import nn

from .hifigan import HifiganGenerator
from .hparams import VITSHparams
from .networks import DurationPredictor, PosteriorEncoder, ResidualCouplingBlocks
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

        self.flow = ResidualCouplingBlocks(
            channels=self.hparams.hidden_channels,
            hidden_channels=self.hparams.hidden_channels,
            kernel_size=self.hparams.kernel_size_flow,
            dilation_rate=self.hparams.dilation_rate_flow,
            num_layers=self.hparams.num_layers_flow,
            cond_channels=self.hparams.embedded_language_dim,
        )

        self.duration_predictor = DurationPredictor(
            self.hparams.hidden_channels,
            256,
            3,
            self.hparams.dropout_p_duration_predictor,
            cond_channels=self.hparams.embedded_language_dim,
            language_emb_dim=self.hparams.embedded_language_dim,
        )

        self.vocoder = HifiganGenerator(
            self.hparams.hidden_channels,
            1,
            self.hparams.resblock_type_decoder,
            self.hparams.resblock_dilation_sizes_decoder,
            self.hparams.resblock_kernel_sizes_decoder,
            self.hparams.upsample_kernel_sizes_decoder,
            self.hparams.upsample_initial_channel_decoder,
            self.hparams.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.hparams.embedded_language_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    def forward(self, x):
        return x
