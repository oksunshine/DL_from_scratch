import numpy as np

class hyperparameters:

    def __init__(self, task_name="regr_sin"):

        if task_name == 'regr_sin':
            import data.sin_config as config
        else:
            assert False, "specified task name({}) is not found".format(task_name)

        self.input_data = config.input_data
        self.input_size = config.input_size
        self.batch_size = config.batch_size
        self.correct_data = config.correct_data
        self.correct_data_size = config.correct_data_size
        self.hidden_layers_shape = config.hidden_layers_shape
        self.epoch_size = config.epoch_size
        self.lr = config.lr
