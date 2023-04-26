from neurones import MPNeurone, Perceptron
from activation_functions import Sigmoid, Step, Linear
from networks import Layer

if __name__ == "__main__":
    or_gate = MPNeurone(n=2, theta=1)
    y = or_gate([0, 1])
    # print("y = " + str(y))

    or_gate_2 = Perceptron(n=2, activation_function=Step(), w=[1.0, 1.0], b=-1)
    y = or_gate_2([1, 0])
    # print("y = " + str(y))

    # invalid_or_gate = Perceptron(n=2, activation_function=Step(), w=[0.0, 1.2], b=-1)
    # print("Wrong output for [1, 0] input: ", invalid_or_gate([1, 0]))
    # invalid_or_gate.learn([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1])
    # print("Correct output for [1, 0] input: ", invalid_or_gate([1, 0]))

    # Neural layer
    or_layer = Layer(2, Perceptron(n=2, activation_function=Step(), w=[1.0, 1.0], b=-1))
    and_layer = Layer(
        1, Perceptron(n=2, activation_function=Step(), w=[1.0, 1.0], b=-2)
    )
    or_layer.append(and_layer)

    print(or_layer([[0, 0], [0, 1]]))
    print(and_layer([[1, 1]]))

    print(or_layer.feed_forward([[1, 1], [0, 0]]))
