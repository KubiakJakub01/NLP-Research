from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn


class SummaryMixing(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_proj_hid_dim: int,
        local_proj_out_dim: int,
        summary_hid_dim: int,
        summary_out_dim: int,
        activation: str,
        mode: Literal['mixing', 'avgonly'],
    ):
        """https://arxiv.org/abs/2307.07421"""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.local_proj_hid_dim = local_proj_hid_dim
        self.local_proj_out_dim = local_proj_out_dim
        self.summary_hid_dim = summary_hid_dim
        self.summary_out_dim = summary_out_dim
        self.activation = self._get_activation(activation)()
        self.mode = mode

        self.local_proj = nn.Sequential(
            nn.Linear(self.d_model, self.local_proj_hid_dim),
            self.activation,
        )
        self.summary_local_merging = nn.Linear(
            self.summary_hid_dim + self.local_proj_hid_dim, self.summary_out_dim
        )

        self.local_norm = nn.LayerNorm(self.local_proj_hid_dim)
        self.summary_norm = nn.LayerNorm(self.summary_hid_dim)

        self.summary_proj = nn.Sequential(
            nn.Linear(self.d_model, self.summary_hid_dim),
            self.activation,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Summary Forward Pass

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, d_model)``
            mask: Padding mask tensor of shape ``(batch_size, seq_len)``
        """
        B, T, _ = x.shape
        if mask is not None:
            mask = rearrange(torch.logical_not(mask), 'b t -> b t 1').float()
        else:
            mask = torch.ones(B, T, 1).float()

        out = x
        if self.mode == 'mixing':
            out = self._forward_mixing(x, mask)
        elif self.mode == 'avgonly':
            out = self._forward_avgonly(x, mask)
        return out

    def _forward_mixing(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape

        # Local Projection
        local_summary = self.local_norm(self.local_proj(x) * mask)

        # Time summary
        time_summary = self.summary_proj(x) * mask
        time_summary = self.summary_norm(torch.sum(time_summary, dim=1) / torch.sum(mask, dim=1))
        time_summary = repeat(time_summary, 'b c -> b t c', t=T)

        # Combine local and time summary
        return self.summary_local_merging(torch.cat([local_summary, time_summary], dim=-1))

    def _forward_avgonly(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape

        time_summary = self.summary_proj(x) * mask
        time_summary = torch.sum(time_summary, dim=1) / torch.sum(mask, dim=1)
        time_summary = repeat(time_summary, 'b c -> b t c', t=T)
        return time_summary

    def _get_activation(self, activation: str):
        activation_dict = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
        }
        return activation_dict[activation]
