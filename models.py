import torch
from torch import nn


class StateActionValueNetwork(nn.Module):
    def __init__(self, state_size: int, n_action: int) -> None:
        super().__init__()
        self.n_action = n_action
        self.layers = nn.Sequential(
            nn.Linear(state_size + n_action - 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 1e-3)

    def forward(self, state: torch.Tensor, action: int) -> torch.Tensor:
        action_ = torch.zeros(self.n_action - 1, dtype=torch.float32)
        if action < 6:
            action_[action] = 1.
        input_ = torch.concat((state.flatten(), action_))
        return self.layers(input_)


class CNNStateActionValueNetwork(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

        for layer in self.layers:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
