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


def train_neuron(  # pylint: disable=too-many-positional-arguments
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
