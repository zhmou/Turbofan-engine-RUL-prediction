import torch
from torch import nn


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=17, hidden_size=50)
        self.attenion = attention3d()
        # to be filled
        self.linear = nn.Sequential(
            nn.Linear(in_features=1, out_features=50),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Dropout(p=0.2),

        )
        self.dense_1 = nn.Linear(in_features= 1, out_features=50)

        self.dense_2 = nn.Linear(in_features=50, out_features=10)


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        x = self.lstm(inputs)
        x = self.attenion(x)
        # flatten
        x.view(batch_size, -1)




class attention3d(nn.Module):
    def __init__(self):
        super(attention3d, self).__init__()

    def forward(self, inputs):
