import numpy as np
from structure.dense import dense
from structure.loss_layer import loss_layer
from structure.functions import *

class model:

    def __init__(self, batch_size, input_size, hidden_layer_shape, output_size,
                 activ_func, grad_activ, activ_in_output, grad_activ_output, loss, lr):

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_shape = hidden_layer_shape
        self.hidden_layer_amount = len(hidden_layer_shape)
        self.activ_func = activ_func
        self.grad_activ = grad_activ
        self.activ_in_output = activ_in_output
        self.grad_activ_output = grad_activ_output
        self.loss = loss
        self.lr = lr
        self.build_model()

    def build_model(self):

        self.hidden_layer = []
        i_size = self.input_size
        for l in range(self.hidden_layer_amount): # l is the number of layer
            self.hidden_layer.append(dense(i_size, self.hidden_layer_shape[l], self.lr, self.activ_func, self.grad_activ))
            i_size = self.hidden_layer_shape[l]

        self.loss_layer = loss_layer(i_size, self.output_size, self.lr, identify_func, grad_identify, loss=self.loss)

    def forward(self, x, t):

        input = x
        for i in range(self.hidden_layer_amount):
            self.hidden_layer[i].forward(input)
            input = self.hidden_layer[i].y.copy()

        self.loss_layer.forward(input, t)

    def backward(self, t):

        self.loss_layer.backward(t)
        # gradient y in previous layer is grad_x in current one.
        grad_y = self.loss_layer.grad_x
        for i in reversed(range(self.hidden_layer_amount)):
            self.hidden_layer[i].backward(grad_y)
            grad_y = self.hidden_layer[i].grad_x

    def predict(self):
        pass