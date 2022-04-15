import numpy as np

# standardization

full_data = np.loadtxt(fname='./datasets/CMAPSSData/test_FD004.txt', dtype=np.float32)
prefix = full_data[:, [0, 1]]
inputs = full_data[:, 2:]

eps = 1e-12
mu = np.mean(inputs, axis=0)
sigma = np.std(inputs, axis=0)

standard = (inputs - mu) / (sigma + eps)

output = np.concatenate((prefix, standard), axis=1)
np.savetxt('./datasets/CMAPSSData/test_FD004_standardized.txt', output, fmt='%f')