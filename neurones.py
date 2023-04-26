# Implement a simple MP neuron model

import numpy as np
from activation_functions import ActivationFunction


# Define an MP neurone object
class MPNeurone:
    def __init__(self, n: int, theta: int) -> None:
        self.n = n  # input size
        self.theta = theta  # threshold

    def __call__(self, x: list[int]) -> int:
        if len(x) != self.n:
            raise ValueError("Input size is incorrect!")

        if np.sum(x) >= self.theta:
            return 1
        else:
            return 0

    def __str__(self) -> str:
        return f"[x_1, ..., x_{self.n}] -> ({self.theta}||) -> y [0 or 1]"


class Perceptron:
    def __init__(
        self,
        n: int,
        activation_function: ActivationFunction,
        w=None,
        b=np.random.rand(),
    ) -> None:
        # If no weight array is provided, generate an array of size n with random values
        if w is None:
            w = np.random.rand(n)
        else:
            # Make sure w is a numpy array
            if n != len(w):
                raise ValueError("Input size is incorrect!")
            w = np.array(w)

        self.n = n  # input size
        self.w = w  # weights BUG: Is this generating a random array
        self.b = b  # bias
        self.activation_function = activation_function

    def __call__(self, x):
        if len(x) != self.n:
            raise ValueError("Input size is incorrect!")

        # Make sure x is a numpy array
        x = np.array(x)

        return self.activation_function((x @ self.w.T) + self.b)

    def __str__(self):
        return f"[x_1, ..., x_{self.n} + {self.b}] -> (\sum|{self.activation_function}) -> y [0 or 1]"
