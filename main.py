from neurones import MPNeurone, Perceptron
from activation_functions import Sigmoid, Step, Linear

if __name__ == "__main__":
    or_gate = MPNeurone(n=2, theta=1)
    y = or_gate([0, 1])
    print("y = " + str(y))

    or_gate_2 = Perceptron(n=2, activation_function=Step(), w=[1, 1], b=-1)
    y = or_gate_2([1, 1])
    print("y = " + str(y))
