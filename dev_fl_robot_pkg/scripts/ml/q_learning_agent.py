#!/usr/bin/env python3
import math
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        cell_size=1.0
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.cell_size = cell_size

        self.Q = defaultdict(lambda: [0.0 for _ in actions])

    # ----------------------------
    # STATE DISCRETIZATION
    # ----------------------------
    def discretize_state(self, state):

        x = state[0]
        y = state[1]
        yaw = state[2]

        # ---------- 1. validity check ----------
        if not math.isfinite(x) or not math.isfinite(y):
            return ("INVALID",)

        if not math.isfinite(yaw):
            yaw = 0.0

        # ---------- 2. clamp workspace ----------
        # maze bounds are about [-2.5, 2.5]
        MAX_POS = 3.0
        x = max(-MAX_POS, min(MAX_POS, x))
        y = max(-MAX_POS, min(MAX_POS, y))

        # ---------- 3. discretize ----------
        x_bin = int(round(x / self.cell_size))
        y_bin = int(round(y / self.cell_size))

        yaw_bin = int((yaw + math.pi) / (math.pi / 4))  # 8 bins
        yaw_bin = max(0, min(7, yaw_bin))

        return (x_bin, y_bin, yaw_bin)
    # ----------------------------
    # ACTION SELECTION
    # ----------------------------
    def select_action(self, state):
        state_key = self.discretize_state(state)

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = self.Q[state_key]
        return self.actions[q_values.index(max(q_values))]

    # ----------------------------
    # LEARNING UPDATE
    # ----------------------------
    def update(self, state, action, reward, next_state, done):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        if state_key == ("INVALID",) or next_state_key == ("INVALID",):
            return
        
        best_next_q = 0.0 if done else max(self.Q[next_state_key])

        a_idx = self.actions.index(action)

        self.Q[state_key][a_idx] += self.alpha * (
            reward + self.gamma * best_next_q - self.Q[state_key][a_idx]
        )

    # ----------------------------
    # END OF EPISODE
    # ----------------------------
    def end_episode(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
