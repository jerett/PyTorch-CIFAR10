import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(LinearClassifier, self).__init__()
        flatten_size = 1
        for shape in input_shape:
            flatten_size *= shape
        self.fc = nn.Linear(flatten_size, num_classes, bias=True)
        nn.init.normal_(self.fc.weight, std=1e-3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
