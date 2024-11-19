import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from ...utils import log_info
from .discriminator import VitsDiscriminator
from .hifigan import HifiganGenerator
from .hparams import VITSHparams
from .networks import DurationPredictor, PosteriorEncoder, ResidualCouplingBlocks
from .transformer import TextEncoder
from .utils import maximum_path, rand_segments, segment


class VITS(nn.Module):
    """VITS TTS model

    Paper: https://arxiv.org/pdf/2106.06103.pdf
    """

    def __init__(self, hparams: VITSHparams):
        super().__init__()
        self.hparams = hparams

        self.init_multilingual()

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

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = rearrange(x_mask, 'b 1 t -> b 1 t 1') * rearrange(y_mask, 'b 1 t -> b 1 1 t')
        with torch.inference_mode():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = rearrange(torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]), 'b t -> b t 1')
            logp2 = torch.einsum('klm, kln -> kmn', [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum('klm, kln -> kmn', [m_p * o_scale, z_p])
            logp4 = rearrange(torch.sum(-0.5 * (m_p**2) * o_scale, [1]), 'b t -> b t 1')
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, rearrange(attn_mask, 'b 1 1 t -> b 1 t')).detach()
            attn = rearrange(attn, 'b t t1 -> b 1 t t1')

        # duration predictor
        attn_durations = attn.sum(3)
        attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
        log_durations = self.duration_predictor(
            x.detach() if self.hparams.detach_dp_input else x,
            x_mask,
            g=g.detach() if self.hparams.detach_dp_input and g is not None else g,
            lang_emb=lang_emb.detach()
            if self.hparams.detach_dp_input and lang_emb is not None
            else lang_emb,
        )
        loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(
            x_mask
        )
        outputs['loss_duration'] = loss_duration
        return outputs, attn

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
        outputs: dict[str, torch.Tensor] = {}
        g, lid, _ = self._set_cond_input(aux_input)

        # language embedding
        lang_emb = None
        if self.hparams.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # duration predictor
        outputs, attn = self.forward_mas(
            outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb
        )

        # expand prior
        m_p = torch.einsum('klmn, kjm -> kjn', [attn, m_p])
        logs_p = torch.einsum('klmn, kjm -> kjn', [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(
            z, y_lengths, self.hparams.spec_segment_size, let_short_samples=True, pad_short=True
        )

        # waveform decoder
        o = self.vocoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.hparams.hop_length,
            self.hparams.spec_segment_size * self.hparams.hop_length,
            pad_short=True,
        )

        outputs.update(
            {
                'model_outputs': o,
                'alignments': rearrange(attn, 'b 1 t t1 -> b t t1'),
                'm_p': m_p,
                'logs_p': logs_p,
                'z': z,
                'z_p': z_p,
                'm_q': m_q,
                'logs_q': logs_q,
                'waveform_seg': wav_seg,
                'slice_ids': slice_ids,
            }
        )

        return outputs

    @torch.inference_mode()
    def inference(
        self,
        x: torch.Tensor,
        aux_input: dict | None = None,
    ):
        """Inference method for the model.

        Args:
            x: Batch of input character sequence IDs.
            aux_input: Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - speaker_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
        """
        return {
            'x': x,
            'aux_input': aux_input,
        }

    def init_multilingual(self):
        """Initialize multilingual modules of a model."""
        if self.hparams.use_language_embedding:
            log_info(' > initialization of language-embedding layers.')
            self.emb_l = nn.Embedding(
                self.hparams.num_languages, self.hparams.embedded_language_dim
            )
            torch.nn.init.xavier_uniform_(self.emb_l.weight)

    @staticmethod
    def _set_cond_input(aux_input: dict | None):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        g, lid, durations = None, None, None
        if aux_input is None:
            return g, lid, durations
        if 'd_vectors' in aux_input and aux_input['d_vectors'] is not None:
            g = rearrange(F.normalize(aux_input['d_vectors']), '... -> ... 1')
            if g.ndim == 2:
                g = rearrange(g, 'b 1 -> 1 b 1')

        if 'language_ids' in aux_input and aux_input['language_ids'] is not None:
            lid = aux_input['language_ids']
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        if 'durations' in aux_input and aux_input['durations'] is not None:
            durations = aux_input['durations']

        return g, lid, durations
