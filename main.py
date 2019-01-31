import numpy as np
from structure.model import model
from data.hyperparam import hyperparameters
from structure.functions import *
from report.plot_result import show_loss_accuracy

def train(model, train_data, label_data, epoch_size):

        loss_log = []
        for i in range(1,epoch_size+1):

                ## forward
                model.forward(train_data, label_data)
                ## backward
                model.backward(label_data)
                loss_log.append(model.loss_layer.loss.mean())
                ## init grad and delta

                # if i % 50 == 0 : show_loss_accuracy(i, loss_log, accuracy=None)

        show_loss_accuracy(i, loss_log, accuracy=None)
        return model


#### define the data
hp = hyperparameters(task_name="regr_sin")

#### build the model
neural_net = model(batch_size = hp.batch_size,
                   input_size = hp.input_size,
                   hidden_layer_shape = hp.hidden_layers_shape,
                   output_size = hp.correct_data_size,
                   activ_func = sigmoid,
                   grad_activ = grad_sigmoid,
                   activ_in_output = activ_in_output,
                   grad_activ_output = grad_activ_in_output,
                   loss_type="mse",
                   lr=hp.lr)

train_model = train(neural_net, hp.input_data, hp.correct_data, 50)

#### train the model

#### predict the data
