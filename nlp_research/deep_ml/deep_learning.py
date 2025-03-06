# pylint: disable=protected-access

import math

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    return z if z > 0 else z * alpha


def sigmoid_derivative(x: np.ndarray | float) -> np.ndarray | float:
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def softmax(scores: list[float]) -> list[float]:
    exp_sum = sum(math.exp(s) for s in scores)
    probabilities = [round(math.exp(s) / exp_sum, 4) for s in scores]
    return probabilities


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> tuple[list[float], float]:
    probabilities: list[float] = []
    for feature in features:
        prob = sum(f_x * weight for f_x, weight in zip(feature, weights, strict=False)) + bias
        probabilities.append(sigmoid(prob))
    mse = sum(
        (label - prob) ** 2 for label, prob in zip(labels, probabilities, strict=False)
    ) / len(labels)
    return probabilities, round(mse, 4)


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    n = labels.shape[0]
    weights = initial_weights
    bias = initial_bias
    mse_values = []

    for _ in range(epochs):
        z = features @ weights + bias
        pred = sigmoid(z)
        loss = round(mse_loss(pred, labels), 4)
        mse_values.append(loss)

        dL_dpred = (2 / n) * (pred - labels)
        dL_dz = dL_dpred * sigmoid_derivative(z)
        dL_dw = features.T @ dL_dz
        dL_db = np.sum(dL_dz)
        weights -= learning_rate * dL_dw
        bias -= learning_rate * dL_db

    return np.round(weights, 4).tolist(), round(bias, 4), mse_values


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()


def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    pad_with = ((padding, padding), (padding, padding))
    input_matrix = np.pad(input_matrix, pad_with, mode='constant', constant_values=0)
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            i_stride = i * stride
            j_stride = j * stride
            output_matrix[i, j] = np.sum(
                input_matrix[
                    i_stride : i_stride + kernel_height, j_stride : j_stride + kernel_width
                ]
                * kernel
            )

    return output_matrix


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    y_pred = X @ w
    loss = np.sum((y_pred - y_true) ** 2) / y_true.shape[0] + alpha * np.sum(w**2)
    return loss


def compute_gradient(X, y, weights):
    """Compute the gradient of MSE loss."""
    m = len(y)
    y_pred = X @ weights
    dL_dpred = (2 / m) * (y_pred - y)
    return X.T @ dL_dpred


def batch_gradient_descent(X, y, weights, learning_rate, n_iterations):
    """Batch Gradient Descent (BGD)"""
    for _ in range(n_iterations):
        gradient = compute_gradient(X, y, weights)
        weights -= learning_rate * gradient
    return weights


def stochastic_gradient_descent(X, y, weights, learning_rate, n_iterations):
    """Stochastic Gradient Descent (SGD)"""
    n = len(y)
    for _ in range(n_iterations):
        indices = np.random.permutation(n)
        for i in indices:
            x_i = X[i : i + 1]
            y_i = y[i : i + 1]
            gradient = compute_gradient(x_i, y_i, weights)
            weights -= learning_rate * gradient
    return weights


def mini_batch_gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size):
    """Mini-Batch Gradient Descent"""
    n = len(y)
    for _ in range(n_iterations):
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_indices = indices[i : i + batch_size]
            x_i = X[batch_indices]
            y_i = y[batch_indices]
            gradient = compute_gradient(x_i, y_i, weights)
            weights -= learning_rate * gradient
    return weights


def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    """Unified function for all gradient descent methods."""
    if method == 'batch':
        return batch_gradient_descent(X, y, weights, learning_rate, n_iterations)
    if method == 'stochastic':
        return stochastic_gradient_descent(X, y, weights, learning_rate, n_iterations)
    if method == 'mini_batch':
        return mini_batch_gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size)
    raise ValueError("Invalid method. Choose 'batch', 'stochastic', or 'mini_batch'.")


def adam_optimizer(
    grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10
):
    m0 = 0
    v0 = 0
    for t in range(1, num_iterations + 1):
        g_t = grad(x0)
        m_t = beta1 * m0 + (1 - beta1) * g_t
        v_t = beta2 * v0 + (1 - beta2) * g_t**2
        mb_t = m_t / (1 - beta1**t)
        vb_t = v_t / (1 - beta2**t)
        x_t = x0 - learning_rate * mb_t / (vb_t ** (1 / 2) + epsilon)

        x0 = x_t.copy()
        m0 = m_t.copy()
        v0 = v_t.copy()

    return x_t


def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V


def self_attention(X, W_q, W_k, W_v):
    def _softmax(x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    d_k = K.shape[1]
    attention_output = _softmax(Q @ K.T / np.sqrt(d_k)) @ V
    return attention_output


def rnn_forward(
    input_sequence: np.ndarray,
    initial_hidden_state: np.ndarray,
    Wx: np.ndarray,
    Wh: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    for x in input_sequence:
        hidden_state = np.tanh(Wx @ x + Wh @ initial_hidden_state + b)
        initial_hidden_state = hidden_state
    return np.round(hidden_state, 4)
