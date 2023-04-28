import numpy as np


class LossFunction:
    def __init__(self, name):
        self.name = name

    def __call__(self, y, y_hat):
        pass

    def derivative(self, y, y_hat):
        pass

    def __str__(self):
        return self.name


class L1(LossFunction):
    def __init__(self):
        super().__init__("L1")

    def __call__(self, y: list, y_hat: list):
        return np.sum(np.abs(np.array(y) - np.array(y_hat)))

    def derivative(self, y, y_hat):
        return np.sign(y_hat - y)


class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__("Mean Squared Error")

    def __call__(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def derivative(self, y, y_hat):
        return 2 * (y_hat - y)
