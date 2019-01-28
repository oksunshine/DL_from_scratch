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
                model.backward()

                ## init grad and delta

                if i % 50 == 0 : print("hello")# show_loss_accuracy(loss, accuracy)

        return model


#### define the data
hp = hyperparameters(task_name="regr_sin")

#### build the model
neural_net = model(hp.batch_size, hp.input_size, hp.hidden_layers_shape, hp.correct_data_size, sigmoid, grad_sigmoid, hp.lr)

train_model = train(neural_net, hp.input_data, hp.correct_data, 100)

#### train the model

#### predict the data
