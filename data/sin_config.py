import numpy as np

input_data = np.arange(0, np.pi*2, 0.1)
correct_data = np.sin(input_data)
input_data = (input_data-np.pi) / np.pi #convert the range of input data (-1.0 ~ 1.0)
input_size = len(input_data)
hidden_layers_shape = [3]
correct_data_size = len(correct_data)
