import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from .models import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CastedSparseEmbedding,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
    trunc_normal_init_,
)


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


class HierarchicalReasoningModel_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModelConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == 'rope':
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == 'learned':
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModelReasoningModule(
            layers=[
                HierarchicalReasoningModelBlock(self.config) for _i in range(self.config.H_layers)
            ]
        )
        self.L_level = HierarchicalReasoningModelReasoningModule(
            layers=[
                HierarchicalReasoningModelBlock(self.config) for _i in range(self.config.L_layers)
            ]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.inference_mode():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input_: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input_.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        # Position embeddings
        if self.config.pos_encodings == 'learned':
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        # Scale
        return self.embed_scale * embedding

    def forward(self):
        pass

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_InnerCarry):
        return HierarchicalReasoningModel_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )
