import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=100_000,
        batch_size=64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory = deque(maxlen=buffer_size)

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.steps = 0

    # -----------------------
    # ACTION SELECTION
    # -----------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)
        return q_vals.argmax().item()

    # -----------------------
    # STORE TRANSITION
    # -----------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # -----------------------
    # LEARN
    # -----------------------
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q = self.policy_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (~dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # -----------------------
    # END EPISODE
    # -----------------------
    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -----------------------
    # SYNC TARGET NETWORK
    # -----------------------
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def build_dqn_state(env, state, goal):

    pose = env.controller.get_pose_state()
    if pose is None:
        return None
    x, y, yaw = pose

    if not math.isfinite(yaw):
        yaw = 0.0

    lidar = env.get_lidar_sectors()
    if lidar is None:
        return None

    front, fl, fr, left, right, back, bl, br = lidar

    MAX_RANGE = 3.5  # must match your laser max range

    lidar_vals = np.array([front, fl, fr, left, right], dtype=np.float32)
    lidar_vals = np.clip(lidar_vals, 0.0, MAX_RANGE) / MAX_RANGE

    dx = goal[0] - x
    dy = goal[1] - y

    goal_dist = math.sqrt(dx*dx + dy*dy)
    goal_angle = math.atan2(dy, dx) - yaw

    # normalize angle to [-pi, pi]
    goal_angle = (goal_angle + math.pi) % (2*math.pi) - math.pi

    
    return np.array([
        front / 3.5,
        fl / 3.5,
        fr / 3.5,
        left / 3.5,
        right / 3.5,
        math.cos(goal_angle),     
        math.sin(goal_angle),     
        goal_dist / 5.0           
    ], dtype=np.float32)
