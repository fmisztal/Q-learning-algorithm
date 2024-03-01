import gymnasium as gym
import numpy as np
import math
import sys
from collections import deque
from agent import Agent

def train(env, agent, episodes, window):
    samp_rewards = deque(maxlen=window)
    avg_rewards = deque(maxlen=episodes)
    best_avg_reward = -math.inf

    for episode in range(1, episodes + 1):
        state = env.reset()[0]
        samp_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state)
            samp_reward += reward
            state = next_state
            if done:
                samp_rewards.append(samp_reward)
                break

        if (episode >= 100):
            avg_reward = np.mean(samp_rewards)
            avg_rewards.append(avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        print(f"\rEpisode {episode}/{episodes} || Best average reward {best_avg_reward}", end="")
        sys.stdout.flush()
        if episode == episodes: print('\n')
    return avg_rewards, best_avg_reward


def play(env, agent):
    state = env.reset()[0]
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        state = next_state
        if done:
            break

if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    agent = Agent(learning_rate=0.9, epsilon=1, gamma=0.9)
    avg_rewards, best_avg_reward = train(env, agent, episodes=10000, window=100)

    env = gym.make("Taxi-v3", render_mode='human')
    agent.eps = 0
    agent.eps_min = 0
    play(env, agent)
