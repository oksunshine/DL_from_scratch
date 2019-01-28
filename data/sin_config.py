# Regression task
import numpy as np

input_data = np.arange(0, np.pi*2, 0.1).reshape(-1, 1)
correct_data = np.sin(input_data)
batch_size, input_size = input_data.shape
batch_size, correct_data_size = correct_data.shape # expect 1
input_data = ((input_data-np.pi) / np.pi) # convert the range of input data (-1.0 ~ 1.0)
hidden_layers_shape = [3]

lr = 0.01
epoch_size = 100
