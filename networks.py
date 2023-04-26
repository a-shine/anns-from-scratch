# Network builder


# Make a layer of MPNeurone or Perceptron
class Layer:
    def __init__(self, nb_neurons: int, neuron_instance) -> None:
        self.neurone_instance = neuron_instance
        self.neurones = []

        self.nb_outputs = nb_neurons
        # self.nb_inputs = neuron_instance.n * nb_neurons

        self.next_layer = None

        for _ in range(nb_neurons):
            self.neurones.append(neuron_instance.copy())

    def __call__(self, x):
        # x is a list of input lists for each neuron
        if len(x) != len(self.neurones):
            raise ValueError("Input size is incorrect!")

        y = []

        # print(self.neurones)
        for i in range(len(self.neurones)):
            y_i = self.neurones[i](x[i])
            y.append(y_i)

        return y

    # Append one layer to the next (maybe link list style)
    def append(self, layer):
        # if self.nb_outputs != layer.nb_inputs:
        #     raise ValueError("Input size is incorrect!")
        self.next_layer = layer

    def feed_forward(self, x):
        # x is a list of input lists for each neuron
        y = self.__call__(x)

        # print(y)
        # print(len(self.next_layer.neurones))

        if self.next_layer is not None:
            return self.next_layer.feed_forward([y])
        else:
            return y

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
