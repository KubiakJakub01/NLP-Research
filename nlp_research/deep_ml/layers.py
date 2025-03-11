import numpy as np

np.random.seed(42)


class Dense:
    def __init__(self, n_units, optimizer, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.optimizer = optimizer
        self.trainable = True
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(low=-limit, high=limit, size=(self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self._X = None

    def forward_pass(self, X, training=True):
        self._X = X
        self.trainable = training
        out = X @ self.W + self.w0
        return out

    def backward_pass(self, accum_grad):
        grad_input = accum_grad @ self.W.T
        if self.trainable:
            grad_weights = self._X.T @ accum_grad
            grad_bias = np.sum(accum_grad, axis=0)
            self.W = self.optimizer.update(self.W, grad_weights)
            self.w0 = self.optimizer.update(self.w0, grad_bias)
        return grad_input

    def number_of_parameters(self):
        return self.layer_input * self.input_shape + 1


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        """
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        self.hidden_state = np.zeros((hidden_size, 1))
        self.x_t = []
        self.y_t = []
        self.h_t = []

    def forward(self, input_sequence):
        """
        Forward pass through the RNN for a given sequence of inputs.
        """
        self.y_t = []
        self.h_t = [self.hidden_state.copy()]
        self.x_t = []

        for t in range(input_sequence.shape[0]):
            x_t = input_sequence[t].reshape(-1, 1)
            self.x_t.append(x_t)

            h_t = np.tanh(self.W_xh @ x_t + self.W_hh @ self.h_t[-1] + self.b_h)
            self.hidden_state = h_t
            self.h_t.append(h_t.copy())

            y_t = self.W_hy @ h_t + self.b_y
            self.y_t.append(y_t.copy())

        return np.array(self.y_t).reshape(-1, 1)

    def backward(self, x, y, learning_rate):
        """
        Backpropagation through time to adjust weights based on error gradient.
        """
        sequence_length = x.shape[0]

        # Initialize gradients to zero
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # Gradient of hidden state for next time step (initially zero)
        dh_next = np.zeros_like(self.hidden_state)

        # Backward pass from t=sequence_length-1 to t=0
        for t in range(sequence_length - 1, -1, -1):
            # Compute loss gradient w.r.t. output (MSE derivative)
            dy_t = (self.y_t[t] - y[t].reshape(-1, 1)) * (2 / sequence_length)

            # Gradient w.r.t. W_hy and b_y
            dW_hy += dy_t @ self.h_t[t + 1].T
            db_y += dy_t

            # Gradient w.r.t. hidden state h_t
            dh = self.W_hy.T @ dy_t + dh_next

            # Gradient w.r.t. z_t (pre-activation), using tanh derivative
            dz_t = dh * (1 - self.h_t[t + 1] ** 2)

            # Gradients w.r.t. W_xh, W_hh, and b_h
            dW_xh += dz_t @ self.x_t[t].T
            dW_hh += dz_t @ self.h_t[t].T
            db_h += dz_t

            # Gradient w.r.t. previous hidden state (for next iteration)
            dh_next = self.W_hh.T @ dz_t

        # Update weights and biases
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
