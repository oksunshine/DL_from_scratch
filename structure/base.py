import numpy as np

class basic_layer:

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size
        self.xavier_init()

    def xavier_init(self):
        # check the references
        self.w = np.random.rand(self.input_size, self.output_size)/np.sqrt(self.input_size)
        self.b = np.random.rand(self.output_size)

    def forward(self):
        # define the each layer
        pass

    def backward(self):
        # define the each layer
        pass

    def update(self, lr):
        # lr means learning rate
        self.w += lr * self.grad
        self.b += lr * self.grad