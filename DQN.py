from torch.optim.lr_scheduler import PolynomialLR
from models import CNNStateActionValueNetwork
from typing import Tuple, Dict
from torch.optim import SGD
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int,
                 state_shape: Tuple[int, int]) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.empty((buffer_size, 1, *state_shape),
                                     dtype=np.float32)
        self.action_memory = np.empty(buffer_size, dtype=np.int64)
        self.reward_memory = np.empty(buffer_size, dtype=np.float32)
        self.next_state_memory = np.empty((buffer_size, 1, *state_shape),
                                          dtype=np.float32)
        self.is_done_memory = np.empty(buffer_size, dtype=np.bool_)
        self.memory_ptr = 0
        self.cur_size = 0

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, is_done: bool) -> None:
        self.state_memory[self.memory_ptr] = np.expand_dims(state, 0)
        self.action_memory[self.memory_ptr] = action
        self.reward_memory[self.memory_ptr] = reward
        self.next_state_memory[self.memory_ptr] = np.expand_dims(next_state, 0)
        self.is_done_memory[self.memory_ptr] = is_done
        self.memory_ptr = (self.memory_ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

    def sample(self) -> Dict[str, torch.Tensor]:
        selected_idxs = np.random.choice(self.cur_size, self.batch_size,
                                         False)
        return {
            "states": torch.from_numpy(self.state_memory[selected_idxs]),
            "actions": torch.from_numpy(self.action_memory[selected_idxs]),
            "rewards": torch.from_numpy(self.reward_memory[selected_idxs]),
            "next_states": torch.from_numpy(self.next_state_memory[
                selected_idxs]),
            "is_dones": torch.from_numpy(self.is_done_memory[selected_idxs])
        }


class Agent:
    def __init__(self, state_shape: Tuple[int, int], n_actions: int,
                 buffer_size: int, batch_size: int, alpha: float, gamma: float,
                 epsilon: float, tau: float) -> None:
        assert 0. < alpha < 1.
        assert 0. < gamma < 1.
        assert 0. <= epsilon <= 1.
        assert 0. < tau < 1.
        self.network = CNNStateActionValueNetwork(n_actions)
        self.target_network = CNNStateActionValueNetwork(n_actions)
        self.target_network.load_state_dict(self.network.state_dict())
        self.network_optimizer = SGD(self.network.parameters(), alpha)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, state_shape)
        self.lr_scheduler = PolynomialLR(self.network_optimizer, 100000, 2.)
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau

    def choose_action(self, state: np.ndarray, action_mask: np.ndarray):
        valid_actions = []
        state_action_values = self.network(torch.from_numpy(np.expand_dims(
            np.expand_dims(state, 0), 0)))[0].detach()
        max_ = -np.inf
        best_action = 0
        for i, state_action_value in enumerate(state_action_values):
            if (state_action_value > max_) and action_mask[i]:
                max_ = state_action_value
                best_action = i
            if action_mask[i]:
                valid_actions.append(i)
        return np.random.choice(valid_actions) if \
            np.random.uniform(0., 1.) <= self.epsilon else best_action

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, is_done: bool) -> float:
        self.replay_buffer.store(state, action, reward, next_state, is_done)
        if self.replay_buffer.cur_size >= self.replay_buffer.batch_size:
            self.network_optimizer.zero_grad()
            data = self.replay_buffer.sample()
            states, actions, rewards, next_states, is_dones = (
                data["states"], data["actions"], data["rewards"],
                data["next_states"], data["is_dones"])
            states_action_values = self.network(states).gather(
                1, actions.unsqueeze(0).T)
            loss = torch.nn.functional.mse_loss(
                states_action_values, rewards.view(-1, 1) + self.gamma *
                self.target_network(next_states).max(1)[0].view(-1, 1).detach()
                * ~is_dones.view(-1, 1))
            loss.backward()
            self.network_optimizer.step()
            self.lr_scheduler.step()
            return loss.item()
        return .0

    def update_target_network(self) -> None:
        for target_param, param in zip(self.target_network.parameters(),
                                       self.network.parameters()):
            target_param.data.copy_(self.tau * param + (1. - self.tau) *
                                    target_param)

    def save(self, path_model1: str, path_model2: str) -> None:
        with open(path_model1, "wb") as f:
            torch.save(self.network.state_dict(), f)
        with open(path_model2, "wb") as f:
            torch.save(self.target_network.state_dict(), f)

    def load(self, path_model1: str, path_model2: str) -> None:
        with open(path_model1, "rb") as f:
            self.network.load_state_dict(torch.load(f))
        with open(path_model2, "rb") as f:
            self.target_network.load_state_dict(torch.load(f))
