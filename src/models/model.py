# TODO: convert to CNN
# class MyAwesomeModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.name = "corruptmnist"
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 10)
#
#         # Dropout module with 0.2 drop probability
#         self.dropout = nn.Dropout(p=0.2)
#
#     def forward(self, x):
#         if x.ndim != 3:
#             raise ValueError('Expected input to a 3D tensor')
#         if x.shape[0] != 1 or x.shape[1] != 28 or x.shape[2] != 28:
#             raise ValueError('Expected each sample to have shape [1, 28, 28]')
#
#         # make sure input tensor is flattened
#
#         x = x.view(x.shape[0], -1)
#
#         # Now with dropout
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.dropout(F.relu(self.fc3(x)))
#
#         # output so no dropout here
#         x = F.log_softmax(self.fc4(x), dim=1)
#
#         return x

from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),  # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),  # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),  # [N, 8, 20]
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10)
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have shape 1, 28, 28")
        return self.classifier(self.backbone(x))
