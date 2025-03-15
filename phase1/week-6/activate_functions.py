import math
from abc import ABC, abstractmethod
from enum import Enum


class ActivateFunctionEnum(str, Enum):
    LINEAR = "linear"
    RELU = "ReLU"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


class ActivateFunction(ABC):
    @abstractmethod
    def activate(self, output: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, output: float) -> float:
        raise NotImplementedError


class Linear(ActivateFunction):
    def activate(self, output: float) -> float:
        return output

    def derivative(self, _: float) -> float:
        return 1


class ReLU(ActivateFunction):
    def activate(self, output: float) -> float:
        return max(0, output)

    def derivative(self, output: float) -> float:
        return 1 if output > 0 else 0


class Sigmoid(ActivateFunction):
    def activate(self, output: float) -> float:
        return 1 / (1 + math.exp(-output))

    def derivative(self, output: float) -> float:
        return output * (1 - output)


class Softmax(ActivateFunction):
    def activate(self, outputs: list[float]) -> list[float]:
        max_output = max(outputs)
        exps = [math.exp(output - max_output) for output in outputs]
        exps_sum = sum(exps)
        return [exp / exps_sum for exp in exps]

    def derivative(self, output: float) -> float:
        return output * (1 - output)


linear = Linear()
relu = ReLU()
sigmoid = Sigmoid()
softmax = Softmax()
