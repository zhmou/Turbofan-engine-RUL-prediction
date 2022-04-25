import torch
import numpy as np
import warnings
from torch.utils.data import Dataset


class Turbofandataset(Dataset):
    def __init__(self, mode='train', dataset=None, rul_result=None, handcrafted_features=None):
        self.data = np.loadtxt(fname=dataset, dtype=np.float32)
        self.window_size = 30
        self.sample_num = self.data[-1][0]
        self.length = []
        self.mode = mode
        self.handcrafted_features = np.loadtxt(fname=handcrafted_features, dtype=np.float32)

        # piece-wise linear RUL target function
        '''
        ATTENTION:
            if you changed the value of max_rul, you need to adjust the multiplier of the output result.
            (at the line 77, 78 of train.py)
        '''
        self.max_rul = 130

        if self.mode == 'test' and rul_result is not None:
            self.rul_result = np.loadtxt(fname=rul_result, dtype=np.float32)
        if self.mode == 'test' and rul_result is None:
            raise ValueError('You did not specify the rul_result file path of the testset, '
                             'please check if the parameters you passed in are correct.')
        if self.mode != 'test' and self.mode != 'train':
            raise ValueError('You chose an undefined mode, '
                             'please check if the parameters you passed in are correct.')
        if self.handcrafted_features is None:
            raise ValueError('You did not specify the handcrafted_features file path, '
                             'please check if the parameters you passed in are correct.')
        if self.mode == 'train' and rul_result is not None:
            warnings.warn('This rul_result file will only be used in the test set, '
                          'and the current you selected is training, so the file will be ignored.')

    def max_time_cycle(self, engineID):
        max_time = np.sum(self.data[:, [0]] == engineID)
        return max_time

    def __len__(self):
        if self.mode == 'train':
            total_length = 0
            for ID in range(int(self.sample_num)):
                max_time = self.max_time_cycle(engineID=ID + 1)
                total_length += max_time - self.window_size + 1
                self.length.append(total_length)
            return total_length

        if self.mode == 'test':
            total_length = self.rul_result.shape[0]
            return total_length

    def __getitem__(self, index):
        if self.mode == 'train':
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
                row += self.max_time_cycle(engineID=i + 1)

            row += idx_in_sample
            start_row_idx = row - 1
            end_row_idx = start_row_idx + self.window_size
            rul = self.max_time_cycle(engineID=ID + 1) - self.data[end_row_idx - 1][1]

        if self.mode == 'test':
            end_row_idx = np.where(self.data[:, [0]] == index + 1)[0][-1] + 1
            start_row_idx = end_row_idx - self.window_size
            rul = self.rul_result[index]

        inputs_numpy = self.data[start_row_idx:end_row_idx, [2, 3, 4, 6, 7, 8,
                                                             11, 12, 13, 15, 16,
                                                             17, 18, 19, 21, 24, 25]]
        inputs = torch.from_numpy(inputs_numpy)
        if rul > self.max_rul:
            rul = self.max_rul
        label = torch.Tensor([rul / int(self.max_rul)])
        handcrafted_feature = torch.from_numpy(self.handcrafted_features[index])
        return inputs, handcrafted_feature, label
