import numpy as np
from structure.base import basic_layer
class loss_layer(basic_layer):

    def __init__(self, input_size, output_size):
        super(loss_layer, self).__init__(input_size, output_size)
        self.grad = np.zeros([input_size, output_size])

    def forward(self, x, label):

        u = np.dot(x, self.w) + self.b
        self.y = u.copy()
        self.loss = 1.0 / 2.0 * (self.y - label)**2
        print("loss : {}".format(self.loss.mean()))

    def backward(self):
        ### calculate the delta
        ### calculate the
        self.grad_y = None

