import torch
import numpy as np


class Trainer:
    def __init__(self, model, model_optimizer, print_every, epochs=200, device='cpu', prefix='FD001'):
        self.model = model.to(device)
        self.model_optimizer = model_optimizer
        self.print_every = print_every
        self.epochs = epochs
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.prefix = prefix

    def train_single_epoch(self, dataloader):
        running_loss = 0

        length = len(dataloader)

        for batch_index, data in enumerate(dataloader, 0):
            inputs, handcrafted_feature, labels = data
            inputs, handcrafted_feature, labels = inputs.to(self.device), handcrafted_feature.to(self.device), labels.to(self.device)
            self.model_optimizer.zero_grad()
            predictions = self.model(inputs, handcrafted_feature)
            loss = self.criterion(predictions, labels)
            running_loss += loss.item()
            loss.backward()

            self.model_optimizer.step()

            if (batch_index + 1) % self.print_every == 0:
                print('batch:{}/{}, loss(avg. on {} batches: {}'.format(batch_index + 1,
                                                                        length,
                                                                        self.print_every,
                                                                        running_loss / self.print_every,
                                                                        ))
                running_loss = 0

    def train(self, train_loader, test_loader, iteration):
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch + 1))
            self.model.train()
            self.train_single_epoch(train_loader)
            current_score, current_RMSE = self.test(test_loader)
            if epoch == 0:
                best_score = current_score
                best_RMSE = current_RMSE
            else:
                if current_score < best_score:
                    best_score = current_score
                    self.save_checkpoints(iteration + 1, epoch + 1, 'best_score')
                if current_RMSE < best_RMSE:
                    best_RMSE = current_RMSE
                    self.save_checkpoints(iteration + 1, epoch + 1, 'best_RMSE')
        return float(best_score), float(best_RMSE)

    def save_checkpoints(self, iteration, epoch, which_type):
        state = {
            'iter': iteration,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optim_dict': self.model_optimizer.state_dict()
        }
        torch.save(state, './checkpoints/{}_iteration{}_{}.pth.tar'.format(self.prefix, iteration, which_type))
        print('{}_checkpoints saved successfully!'.format(which_type))

    @staticmethod
    def score(y_true, y_pred):
        score = 0
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        for i in range(len(y_pred)):
            if y_true[i] <= y_pred[i]:
                score = score + np.exp(-(y_true[i] - y_pred[i]) / 10.0) - 1
            else:
                score = score + np.exp((y_true[i] - y_pred[i]) / 13.0) - 1
        return score

    def test(self, test_loader):
        score = 0
        loss = 0
        self.model.eval()
        criterion = torch.nn.MSELoss()
        for batch_index, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, handcrafted_feature, labels = data
                inputs, handcrafted_feature, labels = inputs.to(self.device), handcrafted_feature.to(self.device), labels.to(self.device)
                predictions = self.model(inputs, handcrafted_feature)
                '''
                    don't change the multiplier(130 or 150) here unless you changed the value of max_rul in turbofandataset.py
                '''
                score += self.score(labels * 150, predictions * 150)
                loss += criterion(labels * 150, predictions * 150) * len(labels)
        loss = (loss / len(test_loader.dataset)) ** 0.5
        print('test result: score: {}, RMSE: {}'.format(score.item(), loss))
        return score.item(), loss
