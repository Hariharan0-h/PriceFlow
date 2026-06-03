"""
DQN agent — Issues #8, #9, #10.
Thin wrapper: holds the network, replay buffer, epsilon schedule.
Training loop is in train.py.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_network import DQNNetwork


class ReplayBuffer:
    """Circular experience replay buffer — Issue #9."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 2,
        n_actions: int = 7,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self._steps = 0

        self.online_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state)
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        current_q = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1).values
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
