import torch
import numpy as np
from torch import optim as optim
from turbofandataset import Turbofandataset
from torch.utils.data import DataLoader
from model import Model
from train import Trainer

if __name__ == '__main__':
    trainset = Turbofandataset(mode='train',
                               dataset='./datasets/CMAPSSData/train_FD004_normed.txt')
    train_loader = DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = Turbofandataset(mode='test',
                              dataset='./datasets/CMAPSSData/test_FD004_normed.txt',
                              rul_result='./datasets/CMAPSSData/RUL_FD004.txt')
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=2)
    print('dataset load successfully!')

    best_score_list = []
    best_RMSE_list = []
    for iteration in range(2):
        print('---Iteration: {}---'.format(iteration + 1))
        model = Model()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(model=model,
                          model_optimizer=optimizer,
                          print_every=50,
                          epochs=epochs,
                          device=device,
                          prefix='FD004')
        best_score, best_RMSE = trainer.train(train_loader, test_loader, iteration)
        best_score_list.append(best_score)
        best_RMSE_list.append(best_RMSE)

    best_score_list = np.array(best_score_list)
    best_RMSE_list = np.array(best_RMSE_list)
    result = np.concatenate((best_score_list, best_RMSE_list)).reshape(2, 2)
    np.savetxt('./{}_result.txt'.format(trainer.prefix), result, fmt='%.4f')
