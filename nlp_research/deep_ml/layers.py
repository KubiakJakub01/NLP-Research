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
