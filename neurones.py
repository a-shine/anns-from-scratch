# Implement a simple MP neuron model

import numpy as np
from activation_functions import ActivationFunction


# Define an MP neurone object
class MPNeurone:
    def __init__(self, n: int, theta: int) -> None:
        self.n = n  # number of inputs
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
            # check that w is an array of floats
            for i in range(len(w)):
                if not isinstance(w[i], float):
                    raise ValueError("Input array must be an array of floats!")
            w = np.array(w)

        self.n = n  # Number of inputs
        self.w = w  # Weights
        self.b = b  # Bias
        self.activation_function = activation_function

    def __call__(self, x):
        if len(x) != self.n:
            raise ValueError("Input size is incorrect!")

        # Make sure x is a numpy array
        x = np.array(x)

        return self.activation_function((x @ self.w.T) + self.b)

    def __str__(self):
        return f"[x_1, ..., x_{self.n} + {self.b}] -> (\sum|{self.activation_function}) -> y"

    # Implementation of the perceptron learning rule
    def learn(self, test_inputs, labels, epochs=100, learning_rate=0.1):
        if len(test_inputs) != len(labels):
            raise ValueError("Input size is incorrect!")

        for _ in range(epochs):
            # chose a random input and its label
            i = np.random.randint(0, len(test_inputs))
            x = test_inputs[i]
            y = labels[i]

            y_hat = self(x)

            print(f"Input: {x}, Label: {y}, Prediction: {y_hat}")

            print(f"Original w = {self.w}")
            for i in range(self.n):
                print(f"Updating w[{i}] on input {x[i]}")
                # Intuition for delta_w; if y - y_hat == 0, then delta_w = 0
                # Why multiply by x[i] - because we need to have a sense of the direction (polarity) of the input. We need the delta to be in the direction that leads to the correct output
                # BUG: Not got a confident intuition for this yet
                delta_w = (y - y_hat) * x[i] * learning_rate
                self.w[i] = self.w[i] + delta_w

            # Update bias (can be treated as a weight with a constant input of 1)
            delta_b = (y - y_hat) * 1 * learning_rate
            self.b = self.b + delta_b

            print(f"New w = {self.w}")
            print(f"New b = {self.b}")

        print("Finished learning!")
        print(f"Final w = {self.w}")

    def copy(self):
        return Perceptron(self.n, self.activation_function, self.w, self.b)


# An MP neuron is a less general version of a perceptron
class MPPerceptron(Perceptron):
    def __init__(self, n: int, theta: int) -> None:
        super().__init__(n, ActivationFunction.step, np.array([1] * n), -theta)

    def __str__(self) -> str:
        return f"[x_1, ..., x_{self.n}] -> ({self.theta}||) -> y [0 or 1]"
