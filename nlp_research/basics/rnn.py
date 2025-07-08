import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_ih = np.random.randn(hidden_size, input_size)
        self.weights_hh = np.random.randn(hidden_size, hidden_size)
        self.weights_ho = np.random.randn(output_size, hidden_size)
        self.bias_h = np.zeros((hidden_size, 1))
        self.bias_o = np.zeros((output_size, 1))

        self.hidden_state = np.zeros((hidden_size, 1))
        self.output_state = np.zeros((output_size, 1))

        self.output_error = None
        self.output_delta = None
        self.hidden_error = None
        self.hidden_delta = None

    def forward(self, input_tensor):
        self.hidden_state = (
            np.dot(self.weights_ih, input_tensor)
            + np.dot(self.weights_hh, self.hidden_state)
            + self.bias_h
        )
        self.hidden_state = np.tanh(self.hidden_state)
        self.output_state = np.dot(self.weights_ho, self.hidden_state) + self.bias_o
        return self.output_state

    def backward(self, input_tensor, output_tensor, learning_rate):
        self.output_error = output_tensor - self.output_state
        self.output_delta = self.output_error * self.relu_derivative(self.output_state)
        self.hidden_error = np.dot(self.weights_ho.T, self.output_delta)
        self.hidden_delta = self.hidden_error * self.relu_derivative(self.hidden_state)
        self.weights_ho += learning_rate * np.dot(self.output_delta, self.hidden_state.T)
        self.bias_o += learning_rate * self.output_delta
        self.weights_ih += learning_rate * np.dot(self.hidden_delta, input_tensor.T)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def train(self, input_tensor, output_tensor, learning_rate, epochs):
        for _ in range(epochs):
            self.backward(input_tensor, output_tensor, learning_rate)

    def predict(self, input_tensor):
        return self.forward(input_tensor)

    def save(self, filename):
        np.savez(
            filename,
            weights_ih=self.weights_ih,
            weights_hh=self.weights_hh,
            weights_ho=self.weights_ho,
            bias_h=self.bias_h,
            bias_o=self.bias_o,
        )

    def load(self, filename):
        data = np.load(filename)
        self.weights_ih = data['weights_ih']


def main():
    rnn = RNN(10, 10, 10)
    rnn.train(np.random.randn(10, 10), np.random.randn(10, 10), 0.1, 100)
    print(rnn.predict(np.random.randn(10, 10)))


if __name__ == '__main__':
    main()
