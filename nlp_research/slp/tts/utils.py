import torch
from einops import rearrange


# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length: torch.Tensor, max_len: int | None = None):
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length: Sequence lengths.
        max_len: Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = int(sequence_length.max().item())
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    return rearrange(seq_range, 't -> 1 t') < rearrange(sequence_length, 'b -> b 1')
