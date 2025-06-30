import numpy as np


def softmax(x):
    """A stable softmax implementation."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calculates the Scaled Dot-Product Attention.

    Args:
        Q (np.array): Queries. Shape: (batch_size, n_queries, d_k)
        K (np.array): Keys. Shape: (batch_size, n_keys, d_k)
        V (np.array): Values. Shape: (batch_size, n_values, d_v)
                      Note: n_keys and n_values must be the same.
        mask (np.array, optional): A boolean mask. Shape: (batch_size, n_queries, n_keys).
                                   Defaults to None.

    Returns:
        tuple: A tuple containing the output and attention weights.
               - output (np.array): The attended-to value vectors. \
                    Shape: (batch_size, n_queries, d_v)
               - attention_weights (np.array): The attention weights. \
                    Shape: (batch_size, n_queries, n_keys)
    """
    # 1. Calculate the dot-product of the Q and K matrices.
    # K.T needs to transpose the last two dimensions for batch matrix multiplication
    # K.shape = (batch_size, n_keys, d_k) -> K.transpose = (batch_size, d_k, n_keys)
    # scores.shape = (batch_size, n_queries, n_keys)
    scores = np.matmul(Q, K.transpose(0, 2, 1))

    # 2. Scale the scores.
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Apply the mask (if provided).
    # The mask is used to prevent the model from "looking" at certain positions.
    # This is crucial for decoding, where you can't see future tokens.
    if mask is not None:
        # We add a very small number (negative infinity) to the masked positions
        # so that they become zero after the softmax.
        scaled_scores = np.where(mask, scaled_scores, -1e9)

    # 4. Apply the softmax function to get attention weights.
    # Softmax is applied on the last axis (the keys dimension).
    attention_weights = softmax(scaled_scores)

    # 5. Multiply the attention weights by the V matrix.
    # output.shape = (batch_size, n_queries, d_v)
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def main():
    # Define dimensions for our simulation
    batch_size = 1
    seq_len = 5  # Length of our input sequence
    d_k = 64  # Dimension of keys/queries

    # Simulate Q, K, and V from the same source (this is "self-attention")
    np.random.seed(42)
    input_sequence = np.random.randn(batch_size, seq_len, d_k)
    Q = input_sequence
    K = input_sequence
    V = input_sequence  # In a real Transformer, Q, K, V are linear projections of the input

    # --- Scenario 1: No Mask ---
    print('--- Scenario 1: No Mask ---')
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print(f'Shape of Output: {output.shape}')
    print(f'Shape of Attention Weights: {attention_weights.shape}')
    print('\nAttention Weights (Sample 0, no mask):')
    print(attention_weights[0].round(3))
    # Note: The diagonal will have high values because a word is most similar to itself.

    # --- Scenario 2: With Mask ---
    # Let's create a "look-ahead" mask for a decoder.
    # It prevents a position from attending to subsequent positions.
    # The mask should be `False` for positions we want to mask (set to -1e9).
    print('\n\n--- Scenario 2: With Look-Ahead Mask ---')
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')  # Upper triangle is True
    # We need to reshape the mask for the batch
    mask = mask.reshape(1, seq_len, seq_len)  # pylint: disable=too-many-function-args

    _, attention_weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)

    print('\nMask:')
    print(mask[0].astype('int'))
    print('\nAttention Weights (Sample 0, with mask):')
    print(attention_weights_masked[0].round(3))


if __name__ == '__main__':
    main()
