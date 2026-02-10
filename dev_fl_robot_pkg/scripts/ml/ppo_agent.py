import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.98
        self.clip_eps = 0.2
        self.lr = 3e-4
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.net = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)

        logits, value = self.net(state_t)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze(0)

    def update(self):
        # --- compute returns ---
        returns = []
        G = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        values = torch.stack(self.values)
        advantages = returns - values.detach()

        # --- PPO update ---
        logits, new_values = self.net(torch.FloatTensor(self.states))
        dist = Categorical(logits=logits)

        new_log_probs = dist.log_prob(torch.LongTensor(self.actions))
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - torch.stack(self.log_probs))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (returns - new_values.squeeze()).pow(2).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_buffer()

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(torch.tensor(log_prob))
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
