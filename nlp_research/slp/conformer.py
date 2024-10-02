import torch.nn as nn
from torchaudio.prototype.models import conformer_rnnt_model


class ConformerRNNTModel(nn.Module):
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
    ):
        super().__init__()
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
            num_symbols=num_symbols,
            symbol_embedding_dim=symbol_embedding_dim,
            num_lstm_layers=num_lstm_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layer_norm=lstm_layer_norm,
            lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
            lstm_dropout=lstm_dropout,
            joiner_activation=joiner_activation,
        )

    def forward(self, x, x_len, y, y_len):
        return self.model(x, x_len, y, y_len)
