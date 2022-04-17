import numpy as np

# normalization

full_data = np.loadtxt(fname='./datasets/CMAPSSData/test_FD004.txt', dtype=np.float32)

# engineID and working time cycle, no need to norm
prefix = full_data[:, [0, 1]]

# operational settings and raw sensor data
inputs = full_data[:, 2:]

# to avoid devide zero
eps = 1e-12

# if you want to try standardiztion:
# mu = np.mean(inputs, axis=0)
# sigma = np.std(inputs, axis=0)
# standard = (inputs - mu) / (sigma + eps)

normed = (inputs - np.min(inputs, axis=0)) / (np.max(inputs, axis=0) - np.min(inputs, axis=0) + eps)

output = np.concatenate((prefix, normed), axis=1)
np.savetxt('./datasets/CMAPSSData/test_FD004_normed.txt', output, fmt='%f')