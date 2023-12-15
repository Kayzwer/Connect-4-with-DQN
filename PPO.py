import torch
import numpy as np
from typing import List, Tuple
from torch.optim import RMSprop
from torch.distributions import Categorical
from models import ActorNetwork, CriticNetwork


class ReplayBuffer:
    def __init__(self, batch_size: int) -> None:
        self.state_memory = []
        self.state_value_memory = []
        self.action_memory = []
        self.action_prob_memory = []
        self.reward_memory = []
        self.is_done_memory = []
        self.batch_size = batch_size

    def store(self, state: np.ndarray, state_value: float, action: int,
              action_prob: float, reward: float, is_done: bool) -> None:
        self.state_memory.append(state)
        self.state_value_memory.append(state_value)
        self.action_memory.append(action)
        self.action_prob_memory.append(action_prob)
        self.reward_memory.append(reward)
        self.is_done_memory.append(is_done)

    def get_data(self, gamma: float, lambda_: float) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.from_numpy(
            np.array(self.state_memory, dtype=np.float32)).unsqueeze(1)
        states_value = torch.from_numpy(
            np.array(self.state_value_memory, dtype=np.float32)).view(-1, 1)
        actions = torch.from_numpy(
            np.array(self.action_memory, dtype=np.int64)).view(-1, 1)
        actions_log_prob = torch.from_numpy(
            np.array(self.action_prob_memory, dtype=np.float32)
        ).view(-1, 1).log()
        is_dones = torch.from_numpy(
            np.array(self.is_done_memory, dtype=np.bool_)).view(-1, 1)

        n = len(self)
        advantages = torch.empty((n, 1), dtype=torch.float32)

        accumulate_gae = 0.
        for i in range(n - 1, -1, -1):
            next_state_value = 0. if (i == n - 1) else states_value[i + 1][0]
            accumulate_gae += gamma * lambda_ * (
                self.reward_memory[i] + gamma * next_state_value * (
                    is_dones[i][0] * 1.) - states_value[i])
            advantages[i][0] = accumulate_gae

        return states, actions, actions_log_prob, advantages

    def generate_batches(self) -> List[np.ndarray]:
        n = len(self.state_memory)
        indexes = np.arange(n, dtype=np.int64)
        np.random.shuffle(indexes)
        return [indexes[i:i + self.batch_size] for i in
                np.arange(0, n, self.batch_size)]

    def clear(self) -> None:
        self.state_memory.clear()
        self.state_value_memory.clear()
        self.action_memory.clear()
        self.action_prob_memory.clear()
        self.reward_memory.clear()
        self.is_done_memory.clear()

    def __len__(self) -> int:
        return len(self.state_value_memory)


class Agent:
    def __init__(self, actor_alpha: float, critic_alpha: float, batch_size: int,
                 gamma: float, lambda_: float, epsilon: float, target_kl: float,
                 n_epochs: int) -> None:
        assert 0. < actor_alpha < 1.
        assert 0. < critic_alpha < 1.
        assert 0 < batch_size
        assert 0. < gamma < 1.
        assert 0. < lambda_ < 1.
        assert 0. < epsilon < 1.
        assert 0. < target_kl < 1.
        self.actor_network = ActorNetwork()
        self.actor_network_optimizer = RMSprop(
            self.actor_network.parameters(), actor_alpha)
        self.critic_netowrk = CriticNetwork()
        self.critic_netowrk_optimizer = RMSprop(
            self.critic_netowrk.parameters(), critic_alpha)
        self.memory = ReplayBuffer(batch_size)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.upper_epsilon = 1. + epsilon
        self.lower_epsilon = 1. - epsilon
        self.target_kl = target_kl
        self.n_epochs = n_epochs

    def choose_action(self, state: np.ndarray, mask: np.ndarray
                      ) -> Tuple[int, float, float]:
        tensor_state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        state_value = self.critic_netowrk(tensor_state)
        action_probs = self.actor_network(tensor_state)[0] * torch.from_numpy(
            mask) * 1.
        action_dist = Categorical(action_probs)
        selected_action = int(action_dist.sample().item())
        return selected_action, action_probs[selected_action].item(), \
            state_value.item()

    def update(self) -> Tuple[float, float]:
        states, actions, actions_log_prob, advantages = self.memory.get_data(
            self.gamma, self.lambda_)

        actor_total_loss = 0.
        critic_total_loss = 0.
        for _ in range(self.n_epochs):
            if (actions_log_prob.exp() * (self.actor_network(
                    states).gather(1, actions).log() -
                    actions_log_prob)).mean() > self.target_kl:
                break
            for batch_index in self.memory.generate_batches():
                new_actions_log_prob = self.actor_network(
                    states[batch_index]).gather(
                    1, actions[batch_index]).log()
                new_states_value = self.critic_netowrk(
                    states[batch_index])
                prob_ratios = (new_actions_log_prob -
                               actions_log_prob[batch_index]).exp()
                surr_loss = prob_ratios * advantages[batch_index]
                clipped_surr_loss = torch.clamp(
                    prob_ratios, self.lower_epsilon, self.upper_epsilon) * \
                    advantages[batch_index]
                actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean()
                critic_loss = torch.nn.functional.mse_loss(
                    new_states_value, advantages[batch_index])
                self.actor_network_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_network_optimizer.step()
                self.critic_netowrk_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_netowrk_optimizer.step()
                actor_total_loss += actor_loss.item()
                critic_total_loss += critic_loss.item()

        self.memory.clear()
        return actor_total_loss, critic_total_loss

    def save(self, actor_path: str, critic_path: str) -> None:
        with open(actor_path, "wb") as f1, open(critic_path, "wb") as f2:
            torch.save(self.actor_network.state_dict(), f1)
            torch.save(self.critic_netowrk.state_dict(), f2)

    def load(self, actor_path: str, critic_path: str) -> None:
        with open(actor_path, "rb") as f1, open(critic_path, "rb") as f2:
            self.actor_network.load_state_dict(torch.load(f1))
            self.critic_netowrk.load_state_dict(torch.load(f2))
