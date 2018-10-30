import numpy as np
from data_loader import CIFAR10Data


def preprocess():
    cifar10_data = CIFAR10Data(train_split=0.9)

    train_data = cifar10_data.X_train
    print(train_data.shape)

    channel_mean = np.mean(train_data, axis=(0, 1, 2)) / 255
    print(channel_mean)
    channel_stdev = np.std(train_data, axis=(0, 1, 2)) / 255
    print(channel_stdev)

    # train_loader, val_loader = cifar10_data.get_dataloader()
    # channel_mean = torch.zeros(3, dtype=torch.float32)
    # train_img_cnt = 0
    # for i, data in enumerate(train_loader, 0):
    #     # get the inputs
    #     inputs, labels = data
    #     channel_sum = torch.sum(inputs, dim=(0, 2, 3))
    #     channel_mean += channel_sum
    #     train_img_cnt += inputs.shape[0]
    # channel_mean /= train_img_cnt * 32 * 32
    # print('train img cnt:', train_img_cnt)
    # print('train channel mean:', channel_mean)
    #
    # channel_stdev = torch.zeros(3, dtype=torch.float32)
    # train_img_cnt = 0
    # for i, data in enumerate(train_loader, 0):
    #     # get the inputs
    #     inputs, labels = data
    #     var = (inputs - channel_mean.reshape(3, 1, 1)) ** 2
    #     var = torch.sum(var, dim=(0, 2, 3))
    #     # channel_sum = torch.sum(inputs, dim=(0, 2, 3))
    #     channel_stdev += var
    #     train_img_cnt += inputs.shape[0]
    # print('train img cnt:', train_img_cnt)
    # channel_pixel_cnt = train_img_cnt * 32 * 32
    # channel_stdev = channel_stdev / channel_pixel_cnt
    # channel_stdev = torch.sqrt(channel_stdev)
    # print('train channel std:', channel_stdev)


if __name__ == '__main__':
    preprocess()
