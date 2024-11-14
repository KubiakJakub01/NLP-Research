import numpy as np
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


def maximum_path(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode='constant', constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path
