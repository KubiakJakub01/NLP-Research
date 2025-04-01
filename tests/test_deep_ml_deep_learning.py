import numpy as np
import pytest

from nlp_research.deep_ml import (
    rnn_forward,
)


@pytest.mark.parametrize(
    'input_sequence, initial_hidden_state, Wx, Wh, b, expected',
    [
        (
            np.array([[1.0], [2.0], [3.0]]),
            np.array([0.0]),
            np.array([[0.5]]),
            np.array([[0.8]]),
            np.array([0.0]),
            [0.9759],
        ),
        (
            np.array([[0.5], [0.1], [-0.2]]),
            np.array([0.0]),
            np.array([[1.0]]),
            np.array([[0.5]]),
            np.array([0.1]),
            [0.118],
        ),
    ],
)
def test_rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b, expected):
    assert np.allclose(rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b), expected)
