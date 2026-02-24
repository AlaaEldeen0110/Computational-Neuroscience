import numpy as np

inputs = np.array([0.05, 0.10])

b1 = 0.5
b2 = 0.7

weights_input_hidden = np.random.uniform(-0.5, 0.5, (2, 2))

weights_hidden_output = np.random.uniform(-0.5, 0.5, (2, 2))

def tanh(x):
    return np.tanh(x)

net_hidden = np.dot(inputs, weights_input_hidden) + b1
out_hidden = tanh(net_hidden)

net_output = np.dot(out_hidden, weights_hidden_output) + b2
final_output = tanh(net_output)

print("Hidden Layer Outputs (h1, h2):")
print(out_hidden)
print("\nFinal Network Output (o1, o2):")
print(final_output)