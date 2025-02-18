import math


def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return round(result, 4)


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
