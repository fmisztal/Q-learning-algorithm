import numpy as np
import random
from collections import defaultdict

class Agent:
    def __init__(self, learning_rate, epsilon, gamma):
        self.nA = 6
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = epsilon
        self.eps_decay = 0.99
        self.eps_min = 0.005
        self.learning_rate = learning_rate
        self.gamma = gamma

    def select_action(self, state):
        self.eps = max(self.eps*self.eps_decay, self.eps_min)
        if random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state):
        current = self.Q[state][action]
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + (self.gamma * Qsa_next)
        new_value = current + (self.learning_rate * (target - current))
        self.Q[state][action] = new_value