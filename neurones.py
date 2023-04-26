# Implement a simple MP neuron model

import numpy as np
import matplotlib.pyplot as plt

# Define an MP neurone object
class MPNeurone:
    def __init__(self, n: int, theta: int) -> None:
        self.n = n # input size
        self.theta = theta # threshold

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
    def __init__(self, n: int, activation_function, w=None, b=np.random.rand()) -> None:
        # If no weight array is provided, generate an array of size n with randiom values
        if w is None:
            w = np.random.rand(n)
        else:
            w = np.array(w)
        
        # make w a vertical vector
        w = w.reshape((n,1))
        print(w)
        if len(w) != n:
            raise ValueError("Input size and weights do not match!")
        self.n = n # input size
        self.w = w # weights BUG: Is this generating a random array
        self.b = b # bias
        self.activation_function = activation_function

        print(type(self.activation_function))
        print(type(w))

    def __call__(self, x):
        if len(x) != self.n:
            raise ValueError("Input size is incorrect!")
        
        x = np.array(x)
        print(x)
        print(self.w.shape)
        print(x.shape)
        print(np.multiply(x, self.w))
        
        if self.activation_function(np.sum(x * self.w) + self.b) >= 0:
            return 1
        else:
            return 0

    def __str__(self):
        return f"[x_1, ..., x_{self.n} + {self.b}] -> (\sum|{self.activation_function}) -> y [0 or 1]"
    

# What are arrays in numpy (they can be anything?)