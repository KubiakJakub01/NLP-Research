import math

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray | float) -> np.ndarray | float:
    return sigmoid(x) * (1 - sigmoid(x))


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
