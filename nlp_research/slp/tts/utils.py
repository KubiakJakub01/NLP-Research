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


def segment(x: torch.Tensor, segment_indices: torch.Tensor, segment_size=4, pad_short=False):
    """Segment each sample in a batch based on the provided segment indices

    Args:
        x: Input tensor.
        segment_indices: Segment indices.
        segment_size: Expected output segment size.
        pad_short: Pad the end of input tensor with zeros if shorter than the segment size.
    """
    # pad the input tensor if it is shorter than the segment size
    if pad_short and x.shape[-1] < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - x.size(2)))

    segments = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        index_start = segment_indices[i]
        index_end = index_start + segment_size
        x_i = x[i]
        if pad_short and index_end >= x.size(2):
            # pad the sample if it is shorter than the segment size
            x_i = torch.nn.functional.pad(x_i, (0, (index_end + 1) - x.size(2)))
        segments[i] = x_i[:, index_start:index_end]
    return segments


def rand_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor | None = None,
    segment_size=4,
    let_short_samples=False,
    pad_short=False,
):
    """Create random segments based on the input lengths.

    Args:
        x: Input tensor.
        x_lengths: Input lengths.
        segment_size: Expected output segment size.
        let_short_samples: Allow shorter samples than the segment size.
        pad_short: Pad the end of input tensor with zeros if shorter than the segment size.

    Shapes:
        - x: :math:`[B, C, T]`
        - x_lengths: :math:`[B]`
    """
    B, _, T = x.size()
    _x_lenghts = T if x_lengths is None else x_lengths.clone()
    if pad_short and segment_size > T:
        x = torch.nn.functional.pad(x, (0, segment_size - T))
        T = segment_size
    len_diff = _x_lenghts - segment_size
    if let_short_samples:
        _x_lenghts[len_diff < 0] = segment_size
        len_diff = _x_lenghts - segment_size
    else:
        assert all(
            len_diff > 0
        ), f'At least one sample is shorter than the segment size ({segment_size}). \n {_x_lenghts}'
    segment_indices = (torch.rand([B]).type_as(x) * (len_diff + 1)).long()
    ret = segment(x, segment_indices, segment_size, pad_short=pad_short)
    return ret, segment_indices
