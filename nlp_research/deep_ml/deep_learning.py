import math


def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return round(result, 4)


def softmax(scores: list[float]) -> list[float]:
    exp_sum = sum(math.exp(s) for s in scores)
    probabilities = [round(math.exp(s) / exp_sum, 4) for s in scores]
    return probabilities
