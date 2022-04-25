import numpy as np

# normalization

full_data = np.loadtxt(fname='./datasets/CMAPSSData/train_FD004.txt', dtype=np.float32)
full_test = np.loadtxt(fname='./datasets/CMAPSSData/test_FD004.txt', dtype=np.float32)

# engineID and working time cycle, no need to norm
prefix_data = full_data[:, [0, 1]]
prefix_test = full_test[:, [0, 1]]
# operational settings and raw sensor data
inputs_data = full_data[:, 2:]
inputs_test = full_test[:, 2:]
# to avoid devide zero
eps = 1e-12

# if you want to try standardiztion:
# mu = np.mean(inputs, axis=0)
# sigma = np.std(inputs, axis=0)
# standard = (inputs - mu) / (sigma + eps)

min = np.min(inputs_data, axis=0)
max = np.max(inputs_data, axis=0)

normed_data = (inputs_data - min) / (max - min + eps)
normed_test = (inputs_test - min) / (max - min + eps)

output_data = np.concatenate((prefix_data, normed_data), axis=1)
output_test = np.concatenate((prefix_test, normed_test), axis=1)
np.savetxt('./datasets/CMAPSSData/train_FD004_normed.txt', output_data, fmt='%f')
np.savetxt('./datasets/CMAPSSData/test_FD004_normed.txt', output_test, fmt='%f')