import numpy as np

class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def __str__(self):
        return "Sigmoid"
    
class Step:
    def __call__(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def __str__(self):
        return "Step"
    
class Linear:
    def __call__(self, x):
        return x

    def __str__(self):
        return "Linear"