import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

        self.input = None
        self.hidden = None
        self.hidden_activated = None
        self.output = None
        self.output_error = None
        self.output_delta = None
        self.hidden_error = None
        self.hidden_delta = None

    def forward(self, X):
        self.input = X
        self.hidden = np.dot(X, self.weights1) + self.bias1
        self.hidden_activated = self.relu(self.hidden)
        self.output = np.dot(self.hidden_activated, self.weights2) + self.bias2
        return self.output

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def backward(self, X, y, output, learning_rate):
        self.output_error = y - output
        self.output_delta = self.output_error * self.relu_derivative(output)

        self.hidden_error = self.output_delta.dot(self.weights2.T)
        self.hidden_delta = self.hidden_error * self.relu_derivative(self.hidden)

        self.weights1 += X.T.dot(self.hidden_delta) * learning_rate
        self.weights2 += self.hidden_activated.T.dot(self.output_delta) * learning_rate
        self.bias1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias2 += np.sum(self.output_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        return self.forward(X)

    def save(self, filename):
        np.savez(
            filename,
            weights1=self.weights1,
            weights2=self.weights2,
            bias1=self.bias1,
            bias2=self.bias2,
        )

    def load(self, filename):
        data = np.load(filename)
        self.weights1 = data['weights1']
        self.weights2 = data['weights2']
        self.bias1 = data['bias1']
        self.bias2 = data['bias2']


def main():
    nn = NeuralNetwork(2, 3, 1)
    nn.train(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]), 1000, 0.1)
    print(nn.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))


if __name__ == '__main__':
    main()
