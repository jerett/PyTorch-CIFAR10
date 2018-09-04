import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from data_loader import CIFAR10Data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _print_progress(progress, content, total_hash=50):
    hash_cnt = int(total_hash * progress)
    hash_str = '#' * hash_cnt
    hash_blank = ' ' * (total_hash - hash_cnt)
    sys.stdout.write('\r[{:s}{:s}] {:.2%} {}'.format(hash_str, hash_blank, progress, content))
    sys.stdout.flush()


def plot_history(history):
    """
    plot loss and acc history.
    :param history: train returned history object
    """
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('acc value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


class CIFAR10Solver(object):
    """
    CIFAR10 classifier solver.
    """
    def __init__(self, model, opt, loss_fn):
        if torch.cuda.is_available():
            self.__device = torch.device('cuda:0')
        else:
            self.__device = torch.device('cpu')
        torch.manual_seed(6666)
        torch.cuda.manual_seed(6666)
        self.data = CIFAR10Data(train_split=0.9)
        self.model = model  # type: nn.Module
        self.model.to(self.__device)
        self.opt = opt  # type: optim.SGD
        self.loss_fn = loss_fn  # type: nn.CrossEntropyLoss
        print('train on device:', self.__device)

    def __test(self, data_loader):
        with torch.no_grad():
            test_loss = 0
            total = 0
            correct = 0

            self.model.eval()
            steps = len(data_loader)
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                test_loss += loss.item()
                total += inputs.size(0)
                predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                correct += np.sum(predictions == labels.cpu().numpy())
            test_loss /= steps
            acc = correct / total
            return test_loss, acc

    def __val(self, val_batch_size):
        val_loader = self.data.get_val_loader(val_batch_size)
        return self.__test(val_loader)

    def train(self, epochs, lr_scheduler=None, train_batch_size=128, val_batch_size=128):
        tran_loader = self.data.get_train_loader(train_batch_size)
        steps = len(tran_loader)
        # print_every_step = np.min([steps, print_every_step])

        print('start training. epoch steps:', steps)
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(epochs):  # loop over the dataset multiple times
            total = 0
            correct = 0
            total_loss = 0

            self.model.train()

            print('Epoch: %d/%d, lr:%.2e' % (epoch + 1, epochs, self.opt.param_groups[0]['lr']))
            for i, data in enumerate(tran_loader):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)

                # zero the parameter gradients
                self.opt.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.opt.step()

                total += inputs.size(0)
                predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                correct += np.sum(predictions == labels.cpu().numpy())

                # caculate loss, avg loss, avg acc
                total_loss += loss.item()
                avg_loss = total_loss / (i + 1)
                avg_acc = correct / total
                progress = (i + 1) / steps
                _print_progress(progress, 'loss:{:.2} acc:{:.2%}'.format(avg_loss, avg_acc))
                if i == steps - 1:
                    history['loss'].append(avg_loss)
                    history['acc'].append(avg_acc)
            val_loss, val_acc = self.__val(val_batch_size)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print(' val_loss:{:.2} val_acc:{:.2%}'.format(val_loss, val_acc))
            # step lr scheduler every epoch
            if lr_scheduler:
                lr_scheduler.step(val_loss)


        print('end training')
        return history

    def test(self, test_batch_size=128):
        test_loader = self.data.get_val_loader(test_batch_size)
        return self.__test(test_loader)
