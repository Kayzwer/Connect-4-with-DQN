import torch
from torch import nn
from typing import Tuple
from torchrl.modules import NoisyLinear


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
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DuelingNoisyQNetwork(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(1, 64, 4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2),
            nn.Flatten(1),
            nn.ReLU()
        )

        self.state_value_layers = nn.Sequential(
            NoisyLinear(384, 256),
            nn.ReLU(),
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )

        self.advantage_value_layers = nn.Sequential(
            NoisyLinear(384, 256),
            nn.ReLU(),
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, n_actions)
        )

        for layer in self.feature_layers:
            if isinstance(layer, (NoisyLinear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.state_value_layers:
            if isinstance(layer, (NoisyLinear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.advantage_value_layers:
            if isinstance(layer, (NoisyLinear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        feature = self.feature_layers(x)
        state_value = self.state_value_layers(feature)
        advantage_value = self.advantage_value_layers(feature)
        return state_value + advantage_value - advantage_value.mean(
            dim=1).view(-1, 1)

    def reset_noise(self):
        for layer in self.state_value_layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

        for layer in self.advantage_value_layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class ActorCriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(1, 64, 4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 7),
            nn.Softmax(dim=1)
        )
        self.critic_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        for layer in self.feature_layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        for layer in self.actor_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        for layer in self.critic_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.feature_layers(x)
        return self.actor_layers(feature), self.critic_layers(feature)

    def forward_actor(self, x) -> torch.Tensor:
        return self.actor_layers(self.feature_layers(x))

    def forward_critic(self, x) -> torch.Tensor:
        return self.critic_layers(self.feature_layers(x))


class ActorNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(1, 64, 4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(64, 7),
            nn.Softmax(dim=1)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        for layer in self.feature_layers:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        torch.nn.init.kaiming_normal_(self.critic_layers[0].weight)
        torch.nn.init.zeros_(self.critic_layers[0].bias)

        torch.nn.init.zeros_(self.actor_layers[0].weight)
        torch.nn.init.zeros_(self.actor_layers[0].bias)
        torch.nn.init.zeros_(self.critic_layers[-1].weight)
        torch.nn.init.zeros_(self.critic_layers[-1].bias)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_layer(self.feature_layers(x))

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_layers(self.feature_layers(x))


class CriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        for layer in self.layers:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
