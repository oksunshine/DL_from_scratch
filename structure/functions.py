
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def grad_sigmoid(x):
    return x * (1.0 - x)

def activ_in_output(x):
    return x

def grad_activ_in_output(x):
    return 1.0

def loss_func(y, t):
    return 1.0 / 2.0 * (y - t)**2

def grad_loss(t, y):
    return t - y

def identify_func(x):
    return x

def grad_identify():
    return 1.0