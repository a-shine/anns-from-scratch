# Network builder

from neurones import MPNeurone, Perceptron
from activation_functions import Sigmoid, Step, Linear


# Make a layer of MPNeurone or Perceptron
class Layer:
    def __init__(self, nb_neurons: int, neurone_type=MPNeurone) -> None:
        self.neurone_type = neurone_type
        self.neurones = []
        for i in range(nb_neurons):
            self.neurones.append(neurone_type.copy())

    def __call__(self, x: list[list[int]]):
        if len(x) != len(self.neurones):
            return False
        y = []
        for i in range(self.neurones):
            if len(x[i]) != self.neurone_type.n:
                return False
            y.append(self.neurones[i](x[i]))
        return y

    # Append one layer to the next (maybe link list style)
    def append(self, layer):
        pass

    def __str__(self):
        return (
            f"[x_1, ..., x_{self.n}] -> (\sum|{self.activation_function}) -> y [0 or 1]"
        )


class Network:
    def __init__(self) -> None:
        pass
        # input layer
        # hidden layer
        # output layer
