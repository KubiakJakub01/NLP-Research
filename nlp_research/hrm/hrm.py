from dataclasses import dataclass

import torch
from pydantic import BaseModel
from torch import nn

from .models.layers import Attention, CosSin, SwiGLU, rms_norm


@dataclass
class HierarchicalReasoningModel_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_Carry:
    inner_carry: HierarchicalReasoningModel_InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: dict[str, torch.Tensor]


class HierarchicalReasoningModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = 'bfloat16'


class HierarchicalReasoningModelBlock(nn.Module):
    def __init__(self, config: HierarchicalReasoningModelConfig) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps
        )
        return hidden_states


class HierarchicalReasoningModelReasoningModule(nn.Module):
    def __init__(self, layers: list[HierarchicalReasoningModelBlock]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states
