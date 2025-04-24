import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=20, output_dim=50):  
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size=26):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
