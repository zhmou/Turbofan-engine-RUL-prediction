import torch
import numpy as np
import warnings
from torch.utils.data import Dataset


class Turbofandataset(Dataset):
    def __init__(self, mode='train', dataset=None, rul_result=None):
        self.data = np.loadtxt(fname=dataset, dtype=np.float32)
        self.window_size = 30
        self.sample_num = self.data[-1][0]
        self.length = []
        self.mode = mode

        self.max_rul = 200

        if self.mode == 'test' and rul_result is not None:
            self.rul_result = np.loadtxt(fname=rul_result, dtype=np.float32)
        if self.mode == 'test' and rul_result is None:
            raise ValueError('You did not specify the rul_result file path of the testset, '
                             'please check if the parameters you passed in are correct.')
        if self.mode != 'test' and self.mode != 'train':
            raise ValueError('You chose an undefined mode, '
                             'please check if the parameters you passed in are correct.')
        if self.mode == 'train' and rul_result is not None:
            warnings.warn('This rul_result file will only be used in the test set, '
                          'and the current you selected is training, so the file will be ignored.')

        # if self.mode == 'train':
        #     self.max_rul = np.max(self.data[:, [1]], axis=0)
        #     print(self.max_rul)
        # if self.mode == 'test':
        #     self.max_rul = np.max(self.rul_result, axis=0)
        #     print(self.max_rul)

    def rest_useful_life(self, engineID):
        rul = np.sum(self.data[:, [0]] == engineID)
        return rul

    def __len__(self):
        total_length = 0
        for ID in range(int(self.sample_num)):
            rul = self.rest_useful_life(engineID=ID + 1)
            total_length += rul - self.window_size + 1
            self.length.append(total_length)
        return total_length

    def __getitem__(self, index):
        ID = 0
        row = 0
        idx_in_sample = index + 1
        for i in self.length:
            if index + 1 > i:
                ID += 1
                idx_in_sample = index + 1 - i
            if index + 1 <= i:
                break

        for i in range(ID):
            row += self.rest_useful_life(engineID=i + 1)

        row += idx_in_sample
        start_row_idx = row - 1
        end_row_idx = start_row_idx + self.window_size

        inputs = torch.from_numpy(self.data[start_row_idx:end_row_idx, [2, 3, 4, 6, 7, 8,
                                                                        11, 12, 13, 15, 16,
                                                                        17, 18, 19, 21, 24, 25]])
        if self.mode == 'train':
            rul = self.rest_useful_life(engineID=ID + 1)
            label = torch.Tensor([rul/int(self.max_rul)])
        else:
            label = torch.Tensor([self.rul_result[ID]/int(self.max_rul)])

        return inputs, label
