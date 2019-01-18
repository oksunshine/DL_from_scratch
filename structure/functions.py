
import numpy as np
"""
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
"""

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def grad_sigmoid(x):
    return x * (1.0 - x)
