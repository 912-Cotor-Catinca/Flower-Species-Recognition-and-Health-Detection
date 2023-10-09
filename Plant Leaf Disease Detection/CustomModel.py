import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.max_pool3(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x