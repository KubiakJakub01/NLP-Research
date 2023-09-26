'''Implementation of the Transformer model.'''
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    '''Multi-head attention mechanism.'''

    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        '''Initialize the class.
        Args:
            d_model: The dimensionality of input and output.
            n_heads: The number of heads.
            dropout_rate: Dropout rate.
        '''
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.to_qkv = nn.Linear(d_model, 3 * d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout_rate)

    def forward(self, x, attn_mask: Tensor | None = None, pad_mask: Tensor | None = None):
        '''Forward of the multi-head attention.

        Args:
            x: Input tensor.
            attn_mask: Attention mask.
            pad_mask: Padding mask.

        Returns:
            The output tensor.
        '''
        # (batch_size, seq_len, d_model)
        qkv = self.to_qkv(x)
        # (batch_size, seq_len, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        # (batch_size, seq_len, d_model)
        x, _ = self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=pad_mask)

        return self.dropout(self.to_out(x))


class PositionWiseFeedForward(nn.Module):
    '''Position-wise feed-forward layer.'''

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        '''Initialize the class.
        Args:
            d_model: The dimensionality of input and output.
            d_ff: The dimensionality of the inner layer.
            dropout_rate: Dropout rate.
        '''
        super(PositionWiseFeedForward, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        '''Forward of the position-wise feed-forward layer.

        Args:
            x: Input tensor.

        Returns:
            The output tensor.
        '''
        # (batch_size, seq_len, d_model)
        return self.ff(x)


class EncoderLayer(nn.Module):
    '''Encoder layer.'''

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        '''Initialize the class.
        Args:
            d_model: The dimensionality of input and output.
            n_heads: The number of heads.
            d_ff: The dimensionality of the inner layer.
            dropout_rate: Dropout rate.
        '''
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask: Tensor | None = None, pad_mask: Tensor | None = None):
        '''Forward of the encoder layer.

        Args:
            x: Input tensor.
            attn_mask: Attention mask.
            pad_mask: Padding mask.

        Returns:
            The output tensor.
        '''
        # (batch_size, seq_len, d_model)
        x = x + self.attn(self.ln1(x), attn_mask, pad_mask)
        # (batch_size, seq_len, d_model)
        x = x + self.ff(self.ln2(x))

        return x


class DecoderLayer(nn.Module):
    '''Decoder layer.'''

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        '''Initialize the class.
        Args:
            d_model: The dimensionality of input and output.
            n_heads: The number of heads.
            d_ff: The dimensionality of the inner layer.
            dropout_rate: Dropout rate.
        '''
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.attn1 = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn2 = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, attn_mask: Tensor | None = None, pad_mask: Tensor | None = None):
        '''Forward of the decoder layer.

        Args:
            x: Input tensor.
            enc_out: Output tensor of the encoder.
            attn_mask: Attention mask.
            pad_mask: Padding mask.

        Returns:
            The output tensor.
        '''
        # (batch_size, seq_len, d_model)
        x = x + self.attn1(self.ln1(x), attn_mask, pad_mask)
        # (batch_size, seq_len, d_model)
        x = x + self.attn2(self.ln2(x), enc_out, pad_mask)
        # (batch_size, seq_len, d_model)
        x = x + self.ff(self.ln3(x))

        return x


class Encoder(nn.Module):
    '''Encoder.'''

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout_rate=0.1):
        '''Initialize the class.
        Args:
            n_layers: The number of layers.
            d_model: The dimensionality of input and output.
            n_heads: The number of heads.
            d_ff: The dimensionality of the inner layer.
            dropout_rate: Dropout rate.
        '''
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])

    def forward(self, x, pad_mask: Tensor | None = None):
        '''Forward of the encoder.

        Args:
            x: Input tensor.
            pad_mask: Padding mask.

        Returns:
            The output tensor.
        '''
        # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)

        return x


class Decoder(nn.Module):
    '''Decoder.'''

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout_rate=0.1):
        '''Initialize the class.
        Args:
            n_layers: The number of layers.
            d_model: The dimensionality of input and output.
            n_heads: The number of heads.
            d_ff: The dimensionality of the inner layer.
            dropout_rate: Dropout rate.
        '''
        super(Decoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])

    def forward(self, x, enc_out, pad_mask: Tensor | None = None):
        '''Forward of the decoder.

        Args:
            x: Input tensor.
            enc_out: Output tensor of the encoder.
            pad_mask: Padding mask.

        Returns:
            The output tensor.
        '''
        # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, enc_out, pad_mask=pad_mask)

        return x
