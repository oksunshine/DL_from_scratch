import numpy as np
from data.hyperparam import *
from structure.model import model
from structure.functions import *

def train(model, train_data, label_data, lr, epoch_size):

    for i in range(1,epoch_size+1):
        pass


#### define the data
hp = hyperparameters(data_name="sin")
del hyperparameters

#### build the model
neural_net = model(hp.input_size, hp.hidden_layers_shape, hp.correct_data_size, sigmoid, grad_sigmoid)
del model, sigmoid, grad_sigmoid

#### train the model

#### predict the data
