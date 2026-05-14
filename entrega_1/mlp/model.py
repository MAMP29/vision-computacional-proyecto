import torch.nn as nn
import torch.nn.functional as F


class ModelMLP(nn.Module):
    def __init__(self, num_classes=4):
        super(ModelMLP, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x