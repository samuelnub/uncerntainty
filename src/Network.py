import torch.nn as nn

class Network(nn.Module):
    def __init__(self, inChannels: int, outChannels: int):
        super().__init__()

        self.inChannels = inChannels
        self.outChannels = outChannels

        self.layers = nn.Sequential(
            nn.Linear(inChannels, inChannels * 16),
            nn.LeakyReLU(),
            nn.Linear(inChannels * 16, inChannels * 8),
            nn.LeakyReLU(),
            nn.Linear(inChannels * 8, outChannels * 16),
            nn.LeakyReLU(),
            nn.Linear(outChannels * 16, outChannels * 8),
            nn.BatchNorm1d(outChannels * 8),
            nn.LeakyReLU(),
            nn.Linear(outChannels * 8, outChannels)
        )

    def forward(self, x):
        return self.layers(x)

    

