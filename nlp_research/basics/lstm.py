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


def main():
    lstm = LSTM(input_size=10, hidden_size=10, output_size=10)
    input_tensor = np.random.randn(10, 10, 10)
    output_tensor = lstm.forward(input_tensor)
    print(output_tensor)


if __name__ == '__main__':
    main()
