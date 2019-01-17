
import numpy as np
from base import basic_layer
from functions import *

class dense(basic_layer):

    def __init__(self, input_size, output_size, act_func, grad_act):
        super(dense, self).__init__(input_size, output_size)
        self.grad = np.zeros(output_size, input_size)
        self.activation_func = act_func
        self.grad_activation = grad_act

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = sigmoid(u)

    def backward(self):
        pass

    def dropout(self):
        pass
