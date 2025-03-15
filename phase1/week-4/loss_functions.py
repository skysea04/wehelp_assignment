import math


def mean_squared_error(pred: list[float], label: list[float]) -> float:
    return sum((label[i] - pred[i]) ** 2 for i in range(len(label))) / len(label)


def binary_cross_entropy(pred: list[float], label: list[float]) -> float:
    return -sum(
        label[i] * math.log(pred[i]) + (1 - label[i]) * math.log(1 - pred[i])
        for i in range(len(label))
    )


def categorical_cross_entropy(pred: list[float], label: list[float]) -> float:
    return -sum(label[i] * math.log(pred[i]) for i in range(len(label)))
