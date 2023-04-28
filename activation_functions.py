import numpy as np


# Activation function abstract class
class ActivationFunction:
    def __call__(self, x):
        pass

    def derivative(self, x):
        pass

    def __str__(self):
        pass


class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x) * (1 - self(x))

    def __str__(self):
        return "Sigmoid"


class Step(ActivationFunction):
    def __call__(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def derivative(self, x):
        return 0

    def __str__(self):
        return "Step"


class Linear(ActivationFunction):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1

    def __str__(self):
        return "Linear"


class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    def __str__(self):
        return "ReLU"


class LeakyReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0.01 * x, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def __str__(self):
        return "LeakyReLU"
