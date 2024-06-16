import numpy as np
import random

import gymnasium as gym
import torch

from environment import *

# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

if __name__ == '__main__':

    batch_size = 3

    env = gym.vector.AsyncVectorEnv([
        lambda: HarlowEnv()
        for _ in range(batch_size)
    ])

    obs, info = env.reset()
    print('initial obs:', obs.reshape(-1))

    dones = np.zeros(batch_size, dtype = bool)

    while not all(dones):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(
            'obs:', obs.reshape(-1), '|',
            'action:', action, '|',
            'correct answer:', info['correct_answer'], '|',
            'reward:', reward, '|',
            'done:', done, '|',
        )
        dones = np.logical_or(dones, done)