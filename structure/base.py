import numpy as np

class basic_layer:

    def __init__(self, input_size, output_size, lr):

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.xavier_init()

    def xavier_init(self):
        # check the references
        self.w = np.random.rand(self.input_size, self.output_size)/np.sqrt(self.input_size)
        self.b = np.random.rand(self.output_size)
        # print("(w, b).shape is ({}, {})".format(self.w.shape, self.b.shape))

    def forward(self):
        # define the each layer
        pass

    def backward(self):
        # define the each layer
        pass

    def update(self, grad_w, grad_b):
        # lr means learning rate
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b