import torch
from loader import Turbofandataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    trainset = Turbofandataset(mode='train', dataset='./datasets/CMAPSSData/train_FD001.txt')
    train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = Turbofandataset(mode='test',
                              dataset='./datasets/CMAPSSData/test_FD001.txt',
                              rul_result='./datasets/CMAPSSData/RUL_FD001.txt')
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=2)

    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        print(inputs, labels)
        break

    for batch_idx, data in enumerate(test_loader, 0):
        inputs, labels = data
        print(inputs, labels)
        break
