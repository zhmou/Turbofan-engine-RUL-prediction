import torch
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

    test_loader = DataLoader(dataset=testset, batch_size=248, shuffle=False, num_workers=2)
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
    trainer.train(train_loader, test_loader)
