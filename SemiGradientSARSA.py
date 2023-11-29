from models import StateActionValueNetwork
from typing import Optional
from torch.optim import SGD
import numpy as np
import torch


class Agent:
    def __init__(self, alpha: float, gamma: float, epsilon: float,
                 n_actions: int) -> None:
        assert 0. < alpha < 1.
        assert 0. < gamma < 1.
        assert 0. <= epsilon <= 1.
        assert 0 < n_actions
        self.network = StateActionValueNetwork(42, 7)
        self.target_network = StateActionValueNetwork(42, 7)
        self.update_target_network()
        self.network_optimizer = SGD(self.network.parameters(), alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        valid_actions = []
        best_actions = []
        max_ = np.NINF
        for i in range(self.n_actions):
            if action_mask[i]:
                valid_actions.append(i)
                if max_ < (state_action_value := self.network(torch.from_numpy(
                        state), i).detach().numpy().item()):
                    max_ = state_action_value
                    best_actions.clear()
                    best_actions.append(i)
                elif max_ == state_action_value:
                    best_actions.append(i)
        return np.random.choice(valid_actions if np.random.uniform(0., 1.) <=
                                self.epsilon else best_actions)

    def update(self, terminate: bool, state: np.ndarray, action: int,
               reward: float, next_state: Optional[np.ndarray] = None,
               next_action: Optional[int] = None) -> float:
        self.network_optimizer.zero_grad()
        next_state_action_value = 0. if terminate else self.target_network(
            torch.from_numpy(next_state), next_action).detach()
        loss = torch.square(reward + self.gamma *
                            next_state_action_value - self.network(
                                torch.from_numpy(state), action))
        loss.backward()
        self.network_optimizer.step()
        return loss.item()
    
    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            torch.save(self.network.state_dict(), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.network.load_state_dict(torch.load(f))
            self.update_target_network()
