from dataclasses import dataclass

import torch
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig


@dataclass
class ConformerEncoderLayerStreamingContext:
    """Streaming metadata and state for a `ConformerEncoderLayer`.

    The multi-head attention and Dynamic Chunk Convolution require to save some
    left context that gets inserted as left padding.

    See :class:`.ConvolutionModule` documentation for further details.
    """

    mha_left_context_size: int
    """For this layer, specifies how many frames of inputs should be saved.
    Usually, the same value is used across all layers, but this can be modified.
    """

    mha_left_context: torch.Tensor | None = None
    """Left context to insert at the left of the current chunk as inputs to the
    multi-head attention. It can be `None` (if we're dealing with the first
    chunk) or `<= mha_left_context_size` because for the first few chunks, not
    enough left context may be available to pad.
    """

    dcconv_left_context: torch.Tensor | None = None
    """Left context to insert at the left of the convolution according to the
    Dynamic Chunk Convolution method.

    Unlike `mha_left_context`, here the amount of frames to keep is fixed and
    inferred from the kernel size of the convolution module.
    """


@dataclass
class ConformerEncoderStreamingContext:
    """Streaming metadata and state for a `ConformerEncoder`."""

    dynchunktrain_config: DynChunkTrainConfig
    """Dynamic Chunk Training configuration holding chunk size and context size
    information."""

    layers: list[ConformerEncoderLayerStreamingContext]
    """Streaming metadata and state for each layer of the encoder."""
