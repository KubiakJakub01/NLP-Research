# pylint: disable=protected-access

import math

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


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
    probabilities = []
    for feature in features:
        prob = sum(f_x * weight for f_x, weight in zip(feature, weights, strict=False)) + bias
        probabilities.append(sigmoid(prob))
    mse = sum(
        (label - prob) ** 2 for label, prob in zip(labels, probabilities, strict=False)
    ) / len(labels)
    mse = round(mse, 4)
    return probabilities, mse


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
