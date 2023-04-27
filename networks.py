# Make a layer of MPNeurone or Perceptron
class Layer:
    def __init__(self, nb_neurons: int, neuron_instance) -> None:
        self.neurone_instance = neuron_instance
        self.neurones = []

        self.nb_outputs = nb_neurons
        self.nb_inputs = neuron_instance.n * nb_neurons
        self.nb_weights = self.nb_inputs + 1  # +1 for the bias

        self.prev_layer = None
        self.next_layer = None

        for _ in range(nb_neurons):
            self.neurones.append(neuron_instance.copy())

    def __call__(self, x):
        # x is a list of input lists for each neuron
        if len(x) != self.nb_inputs:
            raise ValueError("Input size is incorrect!")

        y = []

        # Batch the inputs for each neuron
        index = 0
        for i in range(len(self.neurones)):
            y_i = self.neurones[i](x[index : index + self.neurone_instance.n])
            y.append(y_i)
            index += self.neurone_instance.n  # Move the index to the next batch

        return y

    # Append one layer to the next (maybe link list style)
    def append(self, layer):
        # if self.nb_outputs != layer.nb_inputs:
        #     raise ValueError("Input size is incorrect!")
        # Bidirectionally link the layers
        self.next_layer = layer
        self.next_layer.prev_layer = self

    def _feed_forward(self, x):
        # x is a list of input lists for each neuron (i.e. list of lists)
        y = self.__call__(x)

        if self.next_layer is not None:
            return self.next_layer._feed_forward(y)
        else:
            return y

    def learn(self, x, y, epochs=100, learning_rate=0.1):
        # x is a list of input lists for each neuron
        # y is a list of output lists for each neuron

        for _ in range(epochs):
            # compute the error
            for i in range(len(x)):
                y_hat = self._feed_forward(x[i])
                # BUG: what happens if the output layer is more than a single value?
                error = y[i] - y_hat[0]

                # compute the derivative of every weight and bias in this layer with respect to the error and adjust them proportionally
                # do this this recursively for every layer

    def __str__(self):
        return (
            f"[x_1, ..., x_{self.n}] -> (\sum|{self.activation_function}) -> y [0 or 1]"
        )
