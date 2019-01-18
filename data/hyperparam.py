import numpy as np

class hyperparameters:

    def __init__(self, data_name="sin"):

        if data_name == 'sin':
            import data.sin_config as config

        self.input_data = config.input_data
        self.input_size = config.input_size
        self.correct_data = config.correct_data
        self.correct_data_size = config.correct_data_size
        self.hidden_layers_shape = config.hidden_layers_shape
