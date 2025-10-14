"""Adopted from https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/glow_tts/transformer.py"""

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class RelativePositionMultiHeadAttention(nn.Module):
    """Multi-head attention with Relative Positional embedding.
    https://arxiv.org/pdf/1809.04281.pdf

    It learns positional embeddings for a window of neighbours. For keys and values,
    it learns different set of embeddings. Key embeddings are agregated with the attention
    scores and value embeddings are aggregated with the output.

    Note:
        Example with relative attention window size 2

        - input = [a, b, c, d, e]
        - rel_attn_embeddings = [e(t-2), e(t-1), e(t+1), e(t+2)]

        So it learns 4 embedding vectors (in total 8) separately for key and value vectors.

        Considering the input c

        - e(t-2) corresponds to c -> a
        - e(t-2) corresponds to c -> b
        - e(t-2) corresponds to c -> d
        - e(t-2) corresponds to c -> e

        These embeddings are shared among different time steps. So input a, b, d and e also uses
        the same embeddings.

        Embeddings are ignored when the relative window is out of limit for the first and the last
        n items.

    Args:
        channels (int): input and inner layer channels.
        out_channels (int): output channels.
        num_heads (int): number of attention heads.
        rel_attn_window_size (int, optional): relation attention window size.
            If 4, for each time step next and previous 4 time steps are attended.
            If default, relative encoding is disabled and it is a regular transformer.
            Defaults to None.
        heads_share (bool, optional): [description]. Defaults to True.
        dropout_p (float, optional): dropout rate. Defaults to 0..
        input_length (int, optional): intput length for positional encoding. Defaults to None.
        proximal_bias (bool, optional): enable/disable proximal bias as in the paper.
            Defaults to False.
        proximal_init (bool, optional): enable/disable poximal init as in the paper.
            Init key and query layer weights the same. Defaults to False.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_heads: int,
        rel_attn_window_size: int | None = None,
        heads_share: bool = True,
        dropout_p: float = 0.0,
        input_length: int | None = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0, ' [!] channels should be divisible by num_heads.'
        # class attributes
        self.channels = channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.rel_attn_window_size = rel_attn_window_size
        self.heads_share = heads_share
        self.input_length = input_length
        self.proximal_bias = proximal_bias
        self.dropout_p = dropout_p
        self.attn = None

        # query, key, value layers
        self.k_channels = channels // num_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)

        # output layers
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_p)

        # relative positional encoding layers
        if rel_attn_window_size is not None:
            n_heads_rel = 1 if heads_share else num_heads
            rel_stddev = self.k_channels**-0.5
            emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.register_parameter('emb_rel_k', emb_rel_k)
            self.register_parameter('emb_rel_v', emb_rel_v)

        # init layers
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)

        # proximal bias
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)  # type: ignore
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - c: :math:`[B, C, T]`
            - attn_mask: :math:`[B, 1, T, T]`
        """
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        t_s, t_t = key.size(2), query.size(2)
        query = rearrange(
            query, 'b (n_h d_k) t -> b n_h t d_k', n_h=self.num_heads, d_k=self.k_channels
        )
        key = rearrange(
            key, 'b (n_h d_k) t -> b n_h t d_k', n_h=self.num_heads, d_k=self.k_channels
        )
        value = rearrange(
            value, 'b (n_h d_k) t -> b n_h t d_k', n_h=self.num_heads, d_k=self.k_channels
        )
        # compute raw attention scores
        scores = torch.matmul(
            query, rearrange(key, 'b n_h t d_k -> b n_h d_k t') / math.sqrt(self.k_channels)
        )
        # relative positional encoding for scores
        if self.rel_attn_window_size is not None:
            assert t_s == t_t, 'Relative attention is only available for self-attention.'
            # get relative key embeddings
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        # proximan bias
        if self.proximal_bias:
            assert t_s == t_t, 'Proximal bias is only available for self-attention.'
            scores = scores + self._attn_proximity_bias(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        # attention score masking
        if mask is not None:
            # add small value to prevent oor error.
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.input_length is not None:
                block_mask = (
                    torch.ones_like(scores).triu(-1 * self.input_length).tril(self.input_length)
                )
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        # attention score normalization
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        # apply dropout to attention weights
        p_attn = self.dropout(p_attn)
        # compute output
        output = torch.matmul(p_attn, value)
        # relative positional encoding for values
        if self.rel_attn_window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = rearrange(output, 'b n_h t d_k -> b (n_h d_k) t')
        return output, p_attn

    @staticmethod
    def _matmul_with_relative_values(p_attn: torch.Tensor, re: torch.Tensor):
        """
        Args:
            p_attn: attention weights.
            re: relative value embedding vector. (a_(i,j)^V)

        Shapes:
            -p_attn: :math:`[B, H, T, V]`
            -re: :math:`[H or 1, V, D]`
            -logits: :math:`[B, H, T, D]`
        """
        logits = torch.matmul(p_attn, rearrange(re, '1 t c -> 1 1 t c'))
        return logits

    @staticmethod
    def _matmul_with_relative_keys(query: torch.Tensor, re: torch.Tensor):
        """
        Args:
            query: batch of query vectors. (x*W^Q)
            re: relative key embedding vector. (a_(i,j)^K)

        Shapes:
            - query: :math:`[B, H, T, D]`
            - re: :math:`[H or 1, V, D]`
            - logits: :math:`[B, H, T, V]`
        """
        logits = torch.matmul(query, rearrange(re, '1 t c -> 1 1 c t'))
        return logits

    def _get_relative_embeddings(self, relative_embeddings, length):
        """Convert embedding vestors to a tensor of embeddings"""
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.rel_attn_window_size + 1), 0)
        slice_start_position = max((self.rel_attn_window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings, [0, 0, pad_length, pad_length, 0, 0]
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    @staticmethod
    def _relative_position_to_absolute_position(x):
        """Converts tensor from relative to absolute indexing for local attention.
        Shapes:
            x: :math:`[B, C, T, 2 * T - 1]`
        Returns:
            A Tensor of shape :math:`[B, C, T, T]`
        """
        batch, heads, length, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        # Pad extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    @staticmethod
    def _absolute_position_to_relative_position(x):
        """
        Shapes:
            - x: :math:`[B, C, T, T]`
            - ret: :math:`[B, C, T, 2*T-1]`
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    @staticmethod
    def _attn_proximity_bias(length: int) -> torch.Tensor:
        """Produce an attention mask that discourages distant
        attention values.
        Args:
            length: an integer scalar.
        Returns:
            a Tensor with shape :math:`[1, 1, T, T]`
        """
        # L
        r = torch.arange(length, dtype=torch.float32)
        # L x L
        diff = rearrange(r, 'l -> 1 l') - rearrange(r, 'l -> l 1')
        # scale mask values
        diff = -torch.log1p(torch.abs(diff))
        # 1 x 1 x L x L
        return rearrange(diff, 'l1 l2 -> 1 1 l1 l2')
