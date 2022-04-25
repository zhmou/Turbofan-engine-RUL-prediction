"""
if you changed the value of window_size in turbofandataset.py,
you may need to make modifies on turbodataset and re-run this file
"""

from turbofandataset import Turbofandataset
from torch.utils.data import DataLoader
import numpy as np


# feature extraction of two features: mean value and trend coefficient
def fea_extract(data):
    fea = []
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[1]):
        fea.append(np.mean(data[:, i]))
        fea.append(np.polyfit(x.flatten(), data[:, i].flatten(), deg=1)[0])
    return fea


if __name__ == '__main__':
    # trainset = Turbofandataset(mode='train', dataset='./datasets/CMAPSSData/train_FD004_normed.txt')
    # train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False, num_workers=2)

    testset = Turbofandataset(mode='test', dataset='./datasets/CMAPSSData/test_FD004_normed.txt',
                              rul_result='./datasets/CMAPSSData/RUL_FD004.txt')
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=2)

    handcrafted_feature = []
    for batch_index, data in enumerate(test_loader, 0):
        inputs, labels = data
        for input in inputs:
            handcrafted_feature.append(fea_extract(input.numpy()))
    Array = np.array(handcrafted_feature)
    print(Array.shape)
    np.savetxt('./datasets/CMAPSSData/test_FD004_handcrafted_features.txt', Array, fmt='%f')
