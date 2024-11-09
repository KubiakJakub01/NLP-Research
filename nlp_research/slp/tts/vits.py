import torch
from torch import nn

from .discriminator import VitsDiscriminator
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

        if self.hparams.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.hparams.periods_multi_period_discriminator,
                use_spectral_norm=self.hparams.use_spectral_norm_disriminator,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(  # pylint: disable=dangerous-default-value
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        waveform: torch.Tensor,
        aux_input: dict | None = None,
    ):
        """Forward pass of the model.

        Args:
            x: Batch of input character sequence IDs.
            x_lengths: Batch of input character sequence lengths.
            y: Batch of input spectrograms.
            y_lengths: Batch of input spectrogram lengths.
            waveform: Batch of ground truth waveforms per sample.
            aux_input: Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        return {
            'x': x,
            'x_lengths': x_lengths,
            'y': y,
            'y_lengths': y_lengths,
            'waveform': waveform,
            'aux_input': aux_input,
        }
