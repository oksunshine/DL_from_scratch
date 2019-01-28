
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def grad_sigmoid(x):
    return x * (1.0 - x)
