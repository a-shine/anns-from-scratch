from loss_functions import LossFunction, L1
from neurones import Perceptron
import numpy as np


class Layer:
    def __init__(self, nb_neurons: int, neuron_instance: Perceptron) -> None:
        self.neurone_instance = neuron_instance
        self.neurones = []

        # Make an np.array of size nuurone_instance.n * nb_neurons
        self.inputs = np.zeros(neuron_instance.n * nb_neurons)
        self.weights = np.zeros((neuron_instance.n + 1) * nb_neurons)

        self.prev_layer: Layer = None
        self.next_layer: Layer = None

        for _ in range(nb_neurons):
            self.neurones.append(neuron_instance.copy())

    def __call__(self, x):
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

    # Append one layer to the next
    def append(self, layer):
        # the neuron output for every neuron in the current layer needs to be the input for every neuron in the next layer
        self.nb_outputs = layer.nb_inputs

        # Bidirectionally link the layers to allow for forward and backward propagation
        self.next_layer = layer
        self.next_layer.prev_layer = self

    def _feed_forward(self, x):
        y = self.__call__(x)

        if self.next_layer is not None:
            out = []
            # generate the input for the next layer
            for i in range(self.next_layer.nb_inputs):
                out.append(y[i % self.neurone_instance.n])

            return self.next_layer._feed_forward(out)
        else:
            return y

    def _get_weights(self):
        weights = []
        for neuron in self.neurones:
            weights.extend(neuron.w)
        return weights

    def _update_weights(self, weights):
        index = 0
        for neuron in self.neurones:
            neuron.w = weights[index : index + self.neurone_instance.n]
            index += self.neurone_instance.n

    def learn(
        self, x, y, epochs=100, learning_rate=0.1, loss_function: LossFunction = L1()
    ):
        for _ in range(epochs):
            for i in range(len(x)):
                y_hat = self._feed_forward(x[i])

                # compute the loss (remember y[i] and y_hat may be several outputs so they have to be lists)
                loss = loss_function(y[i], y_hat)

                # move to the last layer
                active_layer = self
                while active_layer.next_layer is not None:
                    active_layer = active_layer.next_layer

                # back propagate
                while active_layer.prev_layer is not None:
                    # compute the gradient
                    gradient = active_layer._get_gradient(y_hat, loss)
                    # update the weights
                    active_layer._update_weights(gradient, learning_rate)
                    active_layer = active_layer.prev_layer
