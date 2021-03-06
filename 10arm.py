import random
import functools
import matplotlib.pyplot as plt
import numpy as np
import math



class Action:
    def __init__(self, name, reward_mean, reward_dev):
        self.name = name
        self.reward_func = functools.partial(random.gauss, reward_mean, reward_dev)
        self.total_times_selected = 0
        self.total_reward_received = 0.0
        self.estimated_reward = 0.0

    def update_reward_estimation(self, reward):
        self.total_reward_received += reward
        self.total_times_selected += 1
        self.estimated_reward = self.total_reward_received / self.total_times_selected


def select_action(actions, exploration):
    if random.random() < exploration:
        return random.choice(actions)
    else:
        r = max(actions, key=lambda x: x.estimated_reward)
        return random.choice([j for j in actions if j.estimated_reward == r.estimated_reward])
    

def simulate(num_runs, num_steps):
    # Initial settings for available actions
    reward_dist_mean = [0.2, -0.8, 1.5, 0.4, 1.2, -1.5, -0.2, -1.0, 0.8, -0.5]
    reward_dist_dev = 1.0

    # Simulation status intialization
    run = 0
    avg_reward_per_step = np.zeros((num_runs, num_steps))

    while run < num_runs:
        available_actions = [Action(i, reward_dist_mean[i], reward_dist_dev) for i in range(10)]
        total_reward = 0.0
        step = 0
        exploration = 0.1
        while step < num_steps:
            a = select_action(available_actions, exploration)
            r = a.reward_func()
            a.update_reward_estimation(r)
            total_reward += r
            avg_reward = total_reward / (step + 1)
            avg_reward_per_step[run, step] = avg_reward
            step += 1
            exploration -= 0.0001 # decaying willing to explore

        run += 1

    plt.plot(np.mean(avg_reward_per_step, axis=0))
    plt.show()

if __name__ == "__main__":
    simulate(1000, 1000)
