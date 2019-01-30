import numpy as np
from structure.functions import loss_func, grad_loss
from structure.base import basic_layer
class loss_layer(basic_layer):

    def __init__(self, input_size, output_size, lr, activ_func, grad_activ, loss):
        super(loss_layer, self).__init__(input_size, output_size, lr)
        self.grad = np.zeros([input_size, output_size])
        self.activation_func = activ_func
        self.grad_activation = grad_activ
        if loss == "mse":
            self.loss_func = loss_func
            self.grad_loss = grad_loss
        else:
            assert False, "that function is not found or implemented"

    def forward(self, x, t):

        self.x = x
        u = np.dot(self.x, self.w) + self.b
        self.y = u.copy() # activation_function: y = x
        self.loss = self.loss_func(self.y, t)
        print("loss : {}".format(self.loss.mean()))

    def backward(self, t):

        delta = self.grad_loss(self.y, t) * self.grad_activation()

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = delta
        self.grad_x = np.dot(delta, self.w.T)

        self.update(self.grad_w, self.grad_b)

        ### update

