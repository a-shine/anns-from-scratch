import numpy as np

np.random.seed(0)

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)

# In an object oriented way
X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Layer:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer(4, 5)
layer2 = Layer(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)
