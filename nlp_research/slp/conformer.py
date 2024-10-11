import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.functional import rnnt_loss
from torchaudio.models import Hypothesis, RNNTBeamSearch
from torchaudio.prototype.models import conformer_rnnt_model


class ASRConformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        time_reduction_stride: int,
        conformer_input_dim: int,
        conformer_ffn_dim: int,
        conformer_num_layers: int,
        conformer_num_heads: int,
        conformer_depthwise_conv_kernel_size: int,
        conformer_dropout: float,
        num_symbols: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_hidden_dim: int,
        lstm_layer_norm: int,
        lstm_layer_norm_epsilon: int,
        lstm_dropout: int,
        joiner_activation: str,
        beam_size: int,
    ):
        super().__init__()
        self.blank = num_symbols
        self.beam_size = beam_size

        self.model = conformer_rnnt_model(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            time_reduction_stride=time_reduction_stride,
            conformer_input_dim=conformer_input_dim,
            conformer_ffn_dim=conformer_ffn_dim,
            conformer_num_layers=conformer_num_layers,
            conformer_num_heads=conformer_num_heads,
            conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
            conformer_dropout=conformer_dropout,
            num_symbols=num_symbols + 1,
            symbol_embedding_dim=symbol_embedding_dim,
            num_lstm_layers=num_lstm_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layer_norm=lstm_layer_norm,
            lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
            lstm_dropout=lstm_dropout,
            joiner_activation=joiner_activation,
        )
        self.decoder = RNNTBeamSearch(self.model, blank=self.blank)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, mel: Tensor, mel_len: Tensor, tokens: Tensor, tokens_len: Tensor) -> Tensor:
        """Forward pass of the model

        Args:
            mel: Mel spectrogram of shape (B, T, C)
            mel_len: Length of mel spectrogram of shape (B)
            tokens: Tokens of shape (B, L)
            tokens_len: Length of tokens of shape (B)

        Returns:
            loss: Loss value"""
        assert mel.dim() == 3 and tokens.dim() == 2, 'Input shape must be (B, T, C) and (B, L)'

        prepended_targets = tokens.new_zeros(tokens.size(0), tokens.size(1) + 1)
        prepended_targets[:, 1:] = tokens
        prepended_targets[:, 0] = self.blank
        prepended_targets_len = tokens_len + 1

        out, src_len, _, _ = self.model(mel, mel_len, prepended_targets, prepended_targets_len)
        loss = rnnt_loss(out, tokens, src_len, tokens_len)
        loss = (loss / mel_len.float()).mean()

        return loss

    @torch.inference_mode()
    def inference(self, mel: Tensor) -> list[Hypothesis]:
        """Inference of the model

        Args:
            mel: Mel spectrogram of shape (B, T, C)

        Returns:
            List of hypothesis"""
        assert mel.dim() == 3, 'Input shape must be (B, T, C)'

        mel_len = torch.tensor([mel.size(1)], device=self.device)
        out = self.decoder(mel, mel_len, self.beam_size)
        return out
