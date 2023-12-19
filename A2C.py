import torch
import numpy as np
from typing import Tuple, Dict
from torch.optim import RMSprop
from torch.distributions import Categorical
from models import PureActorNetwork, CriticNetwork


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int,
                 state_shape: Tuple[int, int]) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.empty((buffer_size, 1, *state_shape),
                                     dtype=np.float32)
        self.reward_memory = np.empty(buffer_size, dtype=np.float32)
        self.next_state_memory = np.empty((buffer_size, 1, *state_shape),
                                          dtype=np.float32)
        self.is_done_memory = np.empty(buffer_size, dtype=np.bool_)
        self.memory_ptr = 0
        self.cur_size = 0

    def store(self, state: np.ndarray, reward: float,
              next_state: np.ndarray, is_done: bool) -> None:
        self.state_memory[self.memory_ptr] = np.expand_dims(state, 0)
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
            "rewards": torch.from_numpy(self.reward_memory[selected_idxs]),
            "next_states": torch.from_numpy(self.next_state_memory[
                selected_idxs]),
            "is_dones": torch.from_numpy(self.is_done_memory[selected_idxs])
        }


class Agent:
    def __init__(self, actor_alpha: float, critic_alpha: float, gamma: float,
                 entropy_weight: float, buffer_size: int, batch_size: int,
                 state_shape: Tuple[int, int], tau: float) -> None:
        assert 0. < actor_alpha < 1.
        assert 0. < critic_alpha < 1.
        assert 0. < gamma < 1.
        assert 0. <= entropy_weight <= 1.
        self.actor_network = PureActorNetwork()
        self.critic_network = CriticNetwork()
        self.target_critic_network = CriticNetwork()
        self.actor_optimizer = RMSprop(self.actor_network.parameters(),
                                       actor_alpha)
        self.critic_optimizer = RMSprop(self.critic_network.parameters(),
                                        critic_alpha)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, state_shape)
        self.gamma = gamma
        self.tau = tau
        self.unit_m_tau = 1. - tau
        self.entropy_weight_multiplier = entropy_weight * \
            torch.tensor(1 / 7, dtype=torch.float32)

    def choose_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        action_dist = Categorical(
            self.actor_network(torch.from_numpy(
                state).unsqueeze(0).unsqueeze(0))[0] * torch.from_numpy(
                    mask) * 1.)
        return int(action_dist.sample().item())

    def soft_update_target_network(self) -> None:
        for target_param, param in zip(
                self.target_critic_network.parameters(),
                self.critic_network.parameters()):
            target_param.data.copy_(
                self.tau * param + self.unit_m_tau * target_param)

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, is_done: bool) -> Tuple[float, float]:
        self.replay_buffer.store(state, reward, next_state, is_done)
        if self.replay_buffer.cur_size >= self.replay_buffer.batch_size:
            self.critic_optimizer.zero_grad()
            data = self.replay_buffer.sample()
            states, rewards, next_states, is_dones = (
                data["states"], data["rewards"].view(-1, 1),
                data["next_states"], data["is_dones"].view(-1, 1))
            states_value = self.critic_network(states)
            targets_value = (rewards + self.gamma * self.target_critic_network(
                next_states) * ~is_dones).detach()
            value_loss = torch.nn.functional.mse_loss(
                states_value, targets_value)
            value_loss.backward()
            self.critic_optimizer.step()
            self.soft_update_target_network()

            tensor_state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
            tensor_next_state = torch.from_numpy(next_state
                                                 ).unsqueeze(0).unsqueeze(0)
            state_value = self.critic_network(tensor_state)
            target_value = self.target_critic_network(tensor_next_state)
            self.actor_optimizer.zero_grad()
            action_dist = Categorical(self.actor_network(tensor_state)[0])
            policy_loss = -(action_dist.log_prob(torch.tensor(action)) *
                            (target_value - state_value).detach() +
                            self.entropy_weight_multiplier *
                            action_dist.entropy())
            policy_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                self.actor_network.parameters(), 2.)
            self.actor_optimizer.step()

            return value_loss.item(), policy_loss.item()
        return 0., 0.

    def save(self, actor_path: str, critic_path: str, target_critic_path: str
             ) -> None:
        with open(actor_path, "wb") as f1, open(critic_path, "wb") as f2, open(
                target_critic_path, "wb") as f3:
            torch.save(self.actor_network.state_dict(), f1)
            torch.save(self.critic_network.state_dict(), f2)
            torch.save(self.target_critic_network.state_dict(), f3)

    def load(self, actor_path: str, critic_path: str, target_critic_path: str
             ) -> None:
        with open(actor_path, "rb") as f1, open(critic_path, "rb") as f2, open(
                target_critic_path, "rb") as f3:
            self.actor_network.load_state_dict(torch.load(f1))
            self.critic_network.load_state_dict(torch.load(f2))
            self.target_critic_network.load_state_dict(torch.load(f3))
