import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class EmotionRecModel(nn.Module):
    def __init__(self):
        super(EmotionRecModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(32, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.1),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.1),
            nn.Linear(1024, 7)
        )

    def forward(self, features):
        out = self.layers(features)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


def resnet():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = MLP(2048, 2048, 7)

    return model
