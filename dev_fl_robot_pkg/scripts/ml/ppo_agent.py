import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
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

        return action.item(), log_prob.detach(), value.squeeze(0).detach()

    def update(self):

        returns = []
        G = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)

        states = torch.from_numpy(np.array(self.states, dtype=np.float32))
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(self.values).squeeze().detach()

        advantages = returns - values
        adv_std = advantages.std()
        if adv_std < 1e-6:
            advantages = advantages * 0.0
        else:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # --- PPO epochs ---
        for _ in range(4):

            logits, new_values = self.net(states)
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

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
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

def build_ppo_state(env, goal,num_lidar_sectors):
    pose = env.controller.get_pose_state()
    if pose is None:
        return None

    x, y, yaw = pose
    if not math.isfinite(yaw):
        yaw = 0.0

    lidar = env.get_lidar_sectors(num_sectors=num_lidar_sectors)
    if lidar is None or len(lidar) != num_lidar_sectors:
        return None

    MAX_RANGE = 3.5
    lidar = np.clip(lidar, 0.0, MAX_RANGE) / MAX_RANGE

    dx = goal[0] - x
    dy = goal[1] - y
    goal_dist = math.sqrt(dx*dx + dy*dy)
    goal_dist = min(goal_dist / 5.0, 1.0)

    goal_angle = math.atan2(dy, dx) - yaw
    goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))

    return np.concatenate([
        lidar,
        np.array([
            math.cos(goal_angle),
            math.sin(goal_angle),
            goal_dist
        ], dtype=np.float32)
    ])
