import math
from enum import Enum


def linear(output: float) -> float:
    return output


def ReLU(output: float) -> float:
    return max(0, output)


def sigmoid(output: float) -> float:
    return 1 / (1 + math.exp(-output))


def softmax(outputs: list[float]) -> list[float]:
    max_output = max(outputs)
    exps = [math.exp(output - max_output) for output in outputs]
    exps_sum = sum(exps)
    return [exp / exps_sum for exp in exps]


class OutputLayerActivateFunction(str, Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
