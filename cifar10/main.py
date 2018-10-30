import torch.optim as optim
import torch.nn as nn
import time
from cifar10_solver import count_parameters
from classifiers.resnet import *

if __name__ == '__main__':
    model = resnet20()
    num = count_parameters(model)
    print('resnet20 pn:', num)

    model = resnet32()
    num = count_parameters(model)
    print('resnet32 pn:', num)

    model = resnet56()
    num = count_parameters(model)
    print('resnet56 pn:', num)

    # svm = SVM(input_shape=(3, 32, 32))
    # print(svm)
    # # print(list(svm.parameters()))
    # opt = optim.SGD(svm.parameters(), lr=1e-3, momentum=0)
    # # opt = optim.SGD(svm.parameters(), lr=1.6e-5, momentum=0.9, weight_decay=0)
    # loss_fn = nn.MultiMarginLoss()
    # solver = CIFAR10Solver(svm, opt, loss_fn)
    # t1 = time.time()
    # solver.train(epochs=4, train_batch_size=128)
    # t2 = time.time()
    # print('train time:%f' % (t2 - t1))
    # test_loss, test_acc = solver.test(test_batch_size=128)
    # print('test_loss:{:.2} test_acc:{:.2%}'.format(test_loss, test_acc))