import numpy as np
from structure.model import model
from data.hyperparam import hyperparameters
from structure.functions import sigmoid, grad_sigmoid
from report.plot_result import show_loss_accuracy

def train(model, train_data, label_data, epoch_size):

        for i in range(1,epoch_size+1):

                ## forward
                model.forward(train_data, label_data)
                ## backward
                # model.backward()

                if i % 50 == 0 : print("hello")# show_loss_accuracy(loss, accuracy)

        return model


#### define the data
hp = hyperparameters(data_name="sin")
del hyperparameters

#### build the model
neural_net = model(hp.batch_size, hp.input_size, hp.hidden_layers_shape, hp.correct_data_size, sigmoid, grad_sigmoid, hp.lr)
del model, sigmoid, grad_sigmoid

train_model = train(neural_net, hp.input_data, hp.correct_data, 5)
#### test
# neural_net.forward(hp.input_data)

#### train the model

#### predict the data
