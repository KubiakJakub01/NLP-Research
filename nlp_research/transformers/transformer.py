'''Implementation of the Transformer model.'''
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



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
