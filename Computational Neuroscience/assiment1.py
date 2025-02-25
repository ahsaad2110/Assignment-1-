import numpy as np

def tanh(x):
    return np.tanh(x)# the output of tanh (-1:1)

inputs = np.array([0.6, 0.8])  #example input

w1 = np.random.uniform(-0.5, 0.5, (2, 2))  
w2 = np.random.uniform(-0.5, 0.5, (2, 1))

b1 = np.array([0.5, 0.7])
b2 = np.array([0.5])

hidden_layer_input = np.dot(inputs, w1) + b1
hidden_layer_output = tanh(hidden_layer_input)

final_input = np.dot(hidden_layer_output, w2) + b2
final_output = tanh(final_input)

print("The Result:", final_output)
