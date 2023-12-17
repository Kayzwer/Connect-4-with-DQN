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
                 entropy_weight: float, beta_clone: float, n_actor_epochs: int,
                 n_critic_epochs: int, n_auxiliary_epochs: int,
                 episode_batch_size: int) -> None:
        assert 0. < actor_alpha < 1.
        assert 0. < critic_alpha < 1.
        assert 0 < batch_size
        assert 0. < gamma < 1.
        assert 0. < lambda_ < 1.
        assert 0. < epsilon < 1.
        assert 0. < target_kl < 1.
        assert 0. < entropy_weight < 1.
        assert 0. <= beta_clone <= 1.
        self.actor_network = ActorNetwork()
        self.actor_network_optimizer = RMSprop(
            self.actor_network.parameters(), actor_alpha)
        self.critic_network = CriticNetwork()
        self.critic_network_optimizer = RMSprop(
            self.critic_network.parameters(), critic_alpha)
        self.memory = ReplayBuffer(batch_size)
        self.memories = [ReplayBuffer(batch_size)
                         for _ in range(episode_batch_size)]
        self.batch_size = batch_size
        self.episode_batch_size = episode_batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.upper_epsilon = 1. + epsilon
        self.lower_epsilon = 1. - epsilon
        self.target_kl = target_kl
        self.entropy_multiplier = entropy_weight / torch.tensor(
            7, dtype=torch.float32).log()
        self.beta_clone = beta_clone
        self.n_actor_epochs = n_actor_epochs
        self.n_critic_epochs = n_critic_epochs
        self.n_auxiliary_epochs = n_auxiliary_epochs

    def store(self, episode: int, state: np.ndarray, state_value: float,
              action: int, action_prob: float, reward: float, is_done: bool
              ) -> None:
        self.memories[episode % self.episode_batch_size].store(
            state, state_value, action, action_prob, reward, is_done)

    def choose_action(self, state: np.ndarray, mask: np.ndarray
                      ) -> Tuple[int, float, float]:
        tensor_state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        state_value = self.critic_network(tensor_state)
        action_probs = self.actor_network.forward_actor(tensor_state)[0] * \
            torch.from_numpy(mask) * 1.
        action_dist = Categorical(action_probs)
        selected_action = int(action_dist.sample().item())
        return selected_action, action_probs[selected_action].item(), \
            state_value.item()

    def generate_batches(self, n: int) -> List[np.ndarray]:
        indexes = np.arange(n, dtype=np.int64)
        np.random.shuffle(indexes)
        return [indexes[i:i + self.batch_size] for i in np.arange(
            0, n, self.batch_size)]

    def update(self, episode: int) -> Tuple[float, float]:
        if episode % self.episode_batch_size != self.episode_batch_size - 1:
            return 0., 0.
        total_length = 0
        all_states = []
        all_actions = []
        all_actions_log_prob = []
        all_advantages = []

        for memory in self.memories:
            states, actions, actions_log_prob, advantages = memory.get_data(
                self.gamma, self.lambda_)
            all_states.append(states)
            all_actions.append(actions)
            all_actions_log_prob.append(actions_log_prob)
            all_advantages.append(advantages)
            total_length += len(memory)
            memory.clear()

        states = torch.cat(all_states)
        actions = torch.cat(all_actions)
        actions_log_prob = torch.cat(all_actions_log_prob)
        advantages = torch.cat(all_advantages)

        actor_total_loss = 0.
        critic_total_loss = 0.
        for _ in range(self.n_actor_epochs):
            if (actions_log_prob.exp() * (self.actor_network.forward_actor(
                    states).gather(1, actions).log() -
                    actions_log_prob)).mean() > self.target_kl:
                break
            for batch_index in self.generate_batches(total_length):
                new_actions_log_prob = self.actor_network.forward_actor(
                    states[batch_index]).gather(
                    1, actions[batch_index]).log()
                action_distributions = Categorical(new_actions_log_prob)
                prob_ratios = (new_actions_log_prob -
                               actions_log_prob[batch_index]).exp()
                surr_loss = prob_ratios * advantages[batch_index]
                clipped_surr_loss = torch.clamp(
                    prob_ratios, self.lower_epsilon, self.upper_epsilon) * \
                    advantages[batch_index]
                actor_loss = -(torch.min(surr_loss, clipped_surr_loss) +
                               self.entropy_multiplier *
                               action_distributions.entropy()).mean()
                self.actor_network_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_network_optimizer.step()
                actor_total_loss += actor_loss.item()

        for _ in range(self.n_critic_epochs):
            for batch_index in self.generate_batches(total_length):
                new_states_value = self.critic_network(
                    states[batch_index])
                critic_loss = torch.nn.functional.mse_loss(
                    new_states_value, advantages[batch_index])
                self.critic_network_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_network_optimizer.step()
                critic_total_loss += critic_loss.item()

        old_actions_prob = self.actor_network.forward_actor(states).detach()
        for _ in range(self.n_auxiliary_epochs):
            auxiliary_loss = torch.nn.functional.mse_loss(
                self.actor_network.forward_critic(states), advantages)
            joint_loss = auxiliary_loss + self.beta_clone * \
                torch.nn.functional.kl_div(self.actor_network.forward_actor(
                    states), old_actions_prob, reduction='batchmean')
            self.actor_network_optimizer.zero_grad()
            joint_loss.backward()
            self.actor_network_optimizer.step()

            critic_loss = torch.nn.functional.mse_loss(self.critic_network(
                states), advantages)
            self.critic_network_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_network_optimizer.step()

        self.memory.clear()
        return actor_total_loss, critic_total_loss

    def save(self, actor_path: str, critic_path: str) -> None:
        with open(actor_path, "wb") as f1, open(critic_path, "wb") as f2:
            torch.save(self.actor_network.state_dict(), f1)
            torch.save(self.critic_network.state_dict(), f2)

    def load(self, actor_path: str, critic_path: str) -> None:
        with open(actor_path, "rb") as f1, open(critic_path, "rb") as f2:
            self.actor_network.load_state_dict(torch.load(f1))
            self.critic_network.load_state_dict(torch.load(f2))
