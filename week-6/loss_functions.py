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


class MeanAbsoluteError(LossFunction):
    def get_total_loss(self, outputs: list[float], excepts: list[float]) -> float:
        self.set_output_and_except(outputs, excepts)

        return sum(
            abs(self.excepts[i] - self.outputs[i]) for i in range(len(self.excepts))
        ) / len(self.excepts)

    def get_output_losses(self) -> list[float]:
        losses = []
        length = len(self.excepts)
        for i in range(length):
            if self.outputs[i] > self.excepts[i]:
                losses.append(1 / length)
            elif self.outputs[i] < self.excepts[i]:
                losses.append(-1 / length)
            else:
                losses.append(0)

        return losses


class BinaryCrossEntropy(LossFunction):
    def get_total_loss(self, outputs: list[float], excepts: list[float]) -> float:
        self.set_output_and_except(outputs, excepts)

        epsilon = 1e-15  # 防止 log(0)

        # 確保輸出值在有效範圍內
        clipped_outputs = [max(min(x, 1 - epsilon), epsilon) for x in self.outputs]

        return sum(
            -self.excepts[i] * math.log(clipped_outputs[i])
            - (1 - self.excepts[i]) * math.log(1 - clipped_outputs[i])
            for i in range(len(self.excepts))
        ) / len(self.excepts)

    def get_output_losses(self) -> list[float]:
        epsilon = 1e-15
        clipped_outputs = [max(min(x, 1 - epsilon), epsilon) for x in self.outputs]

        return [
            (
                -self.excepts[i] / clipped_outputs[i]
                + (1 - self.excepts[i]) / (1 - clipped_outputs[i])
            )
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
