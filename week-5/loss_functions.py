import math
from abc import ABC, abstractmethod


class LossFunction(ABC):
    def __init__(self):
        self.outputs = []
        self.excepts = []

    def set_output_and_except(self, outputs: list[float], excepts: list[float]):
        self.outputs = outputs
        self.excepts = excepts

    @abstractmethod
    def get_total_loss(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_output_losses(self) -> list[float]:
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    def get_total_loss(self, outputs: list[float], excepts: list[float]) -> float:
        self.set_output_and_except(outputs, excepts)

        return sum(
            (self.excepts[i] - self.outputs[i]) ** 2 for i in range(len(self.excepts))
        ) / len(self.excepts)

    def get_output_losses(self) -> list[float]:
        return [
            2 * (self.outputs[i] - self.excepts[i]) / len(self.excepts)
            for i in range(len(self.excepts))
        ]


class BinaryCrossEntropy(LossFunction):
    def get_total_loss(self, outputs: list[float], excepts: list[float]) -> float:
        self.set_output_and_except(outputs, excepts)

        return sum(
            -self.excepts[i] * math.log(self.outputs[i])
            - (1 - self.excepts[i]) * math.log(1 - self.outputs[i])
            for i in range(len(self.excepts))
        ) / len(self.excepts)

    def get_output_losses(self) -> list[float]:
        return [
            -self.excepts[i] / self.outputs[i]
            + (1 - self.excepts[i]) / (1 - self.outputs[i])
            for i in range(len(self.excepts))
        ]


class CategoricalCrossEntropy(LossFunction):
    def get_total_loss(self, outputs: list[float], excepts: list[float]) -> float:
        self.set_output_and_except(outputs, excepts)

        return sum(
            -self.excepts[i] * math.log(self.outputs[i])
            for i in range(len(self.excepts))
        )


def mean_squared_error(pred: list[float], label: list[float]) -> float:
    return sum((label[i] - pred[i]) ** 2 for i in range(len(label))) / len(label)


def binary_cross_entropy(pred: list[float], label: list[float]) -> float:
    return -sum(
        label[i] * math.log(pred[i]) + (1 - label[i]) * math.log(1 - pred[i])
        for i in range(len(label))
    )


def categorical_cross_entropy(pred: list[float], label: list[float]) -> float:
    return -sum(label[i] * math.log(pred[i]) for i in range(len(label)))
