from main import *
import matplotlib.pyplot as plt

def test_learning_rate():
    learning_rates = [0.01, 0.05, 0.1, 0.5, 0.9]
    env = gym.make("Taxi-v3")
    for lr in learning_rates:
        agent = Agent(learning_rate=lr, epsilon=1, gamma=1)
        avg_rewards, best_avg_reward = train(env, agent, 10000)
        plt.plot(range(len(avg_rewards)), avg_rewards, label=f'Lr: {lr}')
    plt.xscale('log')
    plt.legend()
    plt.show()

def test_gamma():
    gammas = [0.1, 0.3, 0.6, 0.9, 1]
    env = gym.make("Taxi-v3")
    for gamma in gammas:
        agent = Agent(learning_rate=0.9, epsilon=1, gamma=gamma)
        avg_rewards, best_avg_reward = train(env, agent, 10000)
        plt.plot(range(len(avg_rewards)), avg_rewards, label=f'Gamma: {gamma}')
    plt.xscale('log')
    plt.legend()
    plt.show()

def test_learning_rate_and_gamma():
    learning_rates = [0.01, 0.05, 0.1, 0.5, 0.9]
    gammas = [0.1, 0.3, 0.6, 0.9, 1]
    env = gym.make("Taxi-v3")
    for lr in learning_rates:
        for gamma in gammas:
            agent = Agent(learning_rate=lr, epsilon=1, gamma=gamma)
            avg_rewards, best_avg_reward = train(env, agent, 10000)
            plt.plot(range(len(avg_rewards)), avg_rewards, label=f'Gamma: {gamma}')
        plt.title(f'Gamma vs avg reward for learning rate = {lr}')
        plt.xlabel('Num of episode')
        plt.ylabel('Average reward')
        plt.xscale('log')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # test_learning_rate()
    # test_gamma()
    test_learning_rate_and_gamma()