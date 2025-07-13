import numpy as np


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))

        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

        self.input = None
        self.hidden_state = None
        self.cell_state = None

        self.input_tensor = None
        self.hidden_state_tensor = None
        self.cell_state_tensor = None

        self.forget_gate = None
        self.input_gate = None
        self.cell_gate = None
        self.output_gate = None

    def forward(self, input_tensor):
        self.input = input_tensor
        self.hidden_state = np.zeros((self.hidden_size, 1))
        self.cell_state = np.zeros((self.hidden_size, 1))

        self.input_tensor = np.zeros((self.input_size, 1))
        self.hidden_state_tensor = np.zeros((self.hidden_size, 1))
        self.cell_state_tensor = np.zeros((self.hidden_size, 1))

        for t in range(input_tensor.shape[0]):
            self.input_tensor = input_tensor[t, :, :]
            self.hidden_state_tensor = self.hidden_state
            self.cell_state_tensor = self.cell_state

            self.input_tensor = np.concatenate(
                (self.input_tensor, self.hidden_state_tensor), axis=0
            )

            self.forget_gate = self.sigmoid(np.dot(self.Wf, self.input_tensor) + self.bf)
            self.input_gate = self.sigmoid(np.dot(self.Wi, self.input_tensor) + self.bi)
            self.cell_gate = np.tanh(np.dot(self.Wc, self.input_tensor) + self.bc)
            self.output_gate = self.sigmoid(np.dot(self.Wo, self.input_tensor) + self.bo)

            self.cell_state = self.forget_gate * self.cell_state + self.input_gate * self.cell_gate
            self.hidden_state = self.output_gate * np.tanh(self.cell_state)

        return self.hidden_state

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, output_tensor, learning_rate):
        """
        Backward pass for the LSTM cell.
        Args:
            output_tensor (np.ndarray): The gradient of the loss with respect to the output (dL/dh).
            learning_rate (float): Learning rate for parameter updates.
        Returns:
            None
        """
        # Initialize gradients for all weights and biases
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)

        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        # Compute output layer gradients (if output layer is used)
        # Here, output_tensor is assumed to be dL/dh (gradient w.r.t. last hidden state)
        dh_next = output_tensor
        dc_next = np.zeros_like(self.cell_state)

        # Retrieve last used values from forward pass
        c = self.cell_state
        f = self.forget_gate
        i = self.input_gate
        g = self.cell_gate
        o = self.output_gate
        xh = self.input_tensor  # concatenated input

        # Gradients w.r.t. output gate
        do = dh_next * np.tanh(c)
        do_raw = do * o * (1 - o)  # sigmoid derivative

        # Gradients w.r.t. cell state
        dc = dh_next * o * (1 - np.tanh(c) ** 2) + dc_next

        # Gradients w.r.t. cell gate
        dg = dc * i
        dg_raw = dg * (1 - g**2)  # tanh derivative

        # Gradients w.r.t. input gate
        di = dc * g
        di_raw = di * i * (1 - i)  # sigmoid derivative

        # Gradients w.r.t. forget gate
        df = dc * self.cell_state_tensor
        df_raw = df * f * (1 - f)  # sigmoid derivative

        # Gradients w.r.t. weights and biases
        dWf += np.dot(df_raw, xh.T)
        dWi += np.dot(di_raw, xh.T)
        dWc += np.dot(dg_raw, xh.T)
        dWo += np.dot(do_raw, xh.T)
        dbf += df_raw
        dbi += di_raw
        dbc += dg_raw
        dbo += do_raw

        # Update weights and biases
        self.Wf -= learning_rate * dWf
        self.Wi -= learning_rate * dWi
        self.Wc -= learning_rate * dWc
        self.Wo -= learning_rate * dWo
        self.bf -= learning_rate * dbf
        self.bi -= learning_rate * dbi
        self.bc -= learning_rate * dbc
        self.bo -= learning_rate * dbo

        # Output layer update (if used)
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby


def main():
    lstm = LSTM(input_size=10, hidden_size=10, output_size=10)
    input_tensor = np.random.randn(10, 10, 10)
    output_tensor = lstm.forward(input_tensor)
    print(output_tensor)


if __name__ == '__main__':
    main()
