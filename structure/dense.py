
import numpy as np
from structure.base import basic_layer

class dense(basic_layer):

    def __init__(self, input_size, output_size, lr, activ_func, grad_activ):
        super(dense, self).__init__(input_size, output_size, lr)
        self.grad = np.zeros([output_size, input_size])
        self.activation_func = activ_func
        self.grad_activation = grad_activ

    def forward(self, x):

        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = self.activation_func(u)

    def backward(self, grad_y):

        delta = grad_y * self.grad_activation(self.y)

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = delta
        self.grad_x = np.dot(delta, self.w.T)

        self.update(self.w, self.b)

    def dropout(self):
        pass
