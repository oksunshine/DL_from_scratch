import numpy as np
from structure.base import basic_layer
class loss_layer(basic_layer):

    def __init__(self, input_size, output_size):
        super(loss_layer, self).__init__(input_size, output_size)
        self.grad = np.zeros([input_size, output_size])

    def forward(self, x, label):

        # calculate u
        u = np.dot(x, self.w) + self.b
        print(x.shape)
        print(self.w.shape)
        # calculate y
        self.y = u.copy()
        # calculate the loss
        self.loss = 1.0 / 2.0 * (self.y - label)**2
        print(self.loss.shape)

    def backward(self):
        return self.y

    def loss(self):
        return 1

