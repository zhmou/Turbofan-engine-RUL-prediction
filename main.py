import torch
from torch import optim as optim
from turbofandataset import Turbofandataset
from torch.utils.data import DataLoader
from model import Model
from train import Trainer

if __name__ == '__main__':
    trainset = Turbofandataset(mode='train',
                               dataset='./datasets/CMAPSSData/train_FD001_normed.txt',
                               handcrafted_features='./datasets/CMAPSSData/train_FD001_handcrafted_features.txt')
    train_loader = DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = Turbofandataset(mode='test',
                              dataset='./datasets/CMAPSSData/test_FD001_normed.txt',
                              rul_result='./datasets/CMAPSSData/RUL_FD001.txt',
                              handcrafted_features='./datasets/CMAPSSData/test_FD001_handcrafted_features.txt')

    test_loader = DataLoader(dataset=testset, batch_size=100, shuffle=False, num_workers=2)
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model=model,
                      model_optimizer=optimizer,
                      print_every=25,
                      epochs=epochs,
                      device=device)
    trainer.train(train_loader, test_loader)
