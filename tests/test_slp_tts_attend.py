import pytest
import torch

from nlp_research.slp.tts.attend import RelativePositionMultiHeadAttention


@pytest.mark.parametrize(
    'batch_size, seq_len, channels, out_channels, num_heads',
    [
        (2, 8, 40, 80, 2),
        (4, 16, 80, 80, 4),
        (8, 32, 160, 80, 8),
    ],
)
def test_relative_position_multi_head_attention(
    batch_size, seq_len, channels, out_channels, num_heads
):
    x = torch.randn(batch_size, channels, seq_len)
    c = torch.randn(batch_size, channels, seq_len)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    attention = RelativePositionMultiHeadAttention(channels, out_channels, num_heads)
    out = attention(x, c, mask)
    assert out.size() == (batch_size, out_channels, seq_len)
