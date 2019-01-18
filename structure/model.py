import numpy as np
from structure.dense import dense
from structure.loss_layer import loss_layer

class model:

    def __init__(self, input_size, hidden_layer_shape, output_size, activ_func, grad_activ):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_shape = hidden_layer_shape
        self.hidden_layer_amount = len(hidden_layer_shape)
        self.activ_func = activ_func
        self.grad_activ = grad_activ
        self.build_model()

    def build_model(self):

        self.hidden_layer = []
        i_size = self.input_size
        for l in range(self.hidden_layer_amount): # l is the number of layer
            self.hidden_layer.append(dense(i_size, self.hidden_layer_shape[l], self.activ_func, self.grad_activ))
            i_size = self.hidden_layer_shape[l]

        self.loss_layer = loss_layer(input_size=i_size, output_size=self.output_size)

    def forward(self, x):
        pass

    def backward(self, y):
        pass

    def predict(self):
        pass

    def dropout(self):
        pass