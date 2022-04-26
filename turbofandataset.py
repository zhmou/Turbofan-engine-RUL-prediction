import torch
import numpy as np
import warnings
from torch.utils.data import Dataset


class Turbofandataset(Dataset):
    def __init__(self, mode='train', dataset=None, rul_result=None):
        self.data = np.loadtxt(fname=dataset, dtype=np.float32)
        self.data = np.delete(self.data, [5, 9, 10, 14, 20, 22, 23], axis=1)
        self.window_size = 30
        self.sample_num = int(self.data[-1][0])
        self.length = []
        self.mode = mode

        '''
        piece-wise linear RUL target function

        ATTENTION:
            if you changed the value of max_rul, you need to adjust the multiplier of the output result.
            (at the line 77, 78 of train.py)
        '''
        self.max_rul = 150

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
                          'and the current mode you selected is training, so the file will be ignored.')

        self.x = []
        self.mean_and_coef = []
        self.y = []

        if self.mode == 'train':
            for i in range(1, self.sample_num + 1):
                ind = np.where(self.data[:, 0] == i)
                # transfer tuple to ndarray
                ind = ind[0]
                # single engine data
                data_temp = self.data[ind, :]
                for j in range(len(data_temp) - self.window_size + 1):
                    self.x.append(data_temp[j: j+self.window_size, 2:])
                    rul = len(data_temp) - self.window_size - j
                    if rul > self.max_rul:
                        rul = self.max_rul
                    self.y.append(rul)

        if self.mode == 'test':
            for i in range(1, self.sample_num + 1):
                ind = np.where(self.data[:, 0] == i)[0]
                data_temp = self.data[ind, :]
                '''
                    When the number of data for a turbofan engine on the testset is less than the window length, 
                    an interpolation operation will be performed
                '''
                if len(data_temp) < self.window_size:
                    data = np.zeros((self.window_size, data_temp.shape[1]), dtype=np.float64)
                    for j in range(data.shape[1]):
                        x_old = np.linspace(0, len(data_temp)-1, len(data_temp))
                        params = np.polyfit(x_old, data_temp[:, j].flatten(), deg=1)
                        k = params[0]
                        b = params[1]
                        x_new = np.linspace(0, self.window_size-1, self.window_size, dtype=np.float64)
                        data[:, j] = (x_new * len(data_temp) / self.window_size * k + b)
                    self.x.append(data[-self.window_size:, 2:])
                else:
                    self.x.append(data_temp[-self.window_size:, 2:])
                rul = self.rul_result[i - 1]
                if rul > self.max_rul:
                    rul = self.max_rul
                self.y.append(rul)

        self.x = np.array(self.x)
        self.y = np.array(self.y)/self.max_rul
        for i in range(len(self.x)):
            one_sample = self.x[i]
            self.mean_and_coef.append(self.fea_extract(one_sample))
        self.mean_and_coef = np.array(self.mean_and_coef)

    @staticmethod
    def fea_extract(data):
        fea = []
        x = np.array(range(data.shape[0]))
        for i in range(data.shape[1]):
            fea.append(np.mean(data[:, i]))
            fea.append(np.polyfit(x.flatten(), data[:, i].flatten(), deg=1)[0])
        return fea

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x_tensor = torch.from_numpy(self.x[index]).to(torch.float32)
        y_tensor = torch.Tensor([self.y[index]]).to(torch.float32)
        handcrafted_features = torch.from_numpy(self.mean_and_coef[index]).to(torch.float32)
        return x_tensor, handcrafted_features, y_tensor




