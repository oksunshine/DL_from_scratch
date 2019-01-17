import numpy as np
from base import basic_layer

class loss_layer(basic_layer):

    def __init__(self, input_size, output_size):
        super(loss_layer, self).__init__(input_size, output_size)
        self.grad = np.zeros([input_size, output_size])

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def loss(self):
        return 1

