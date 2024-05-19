import numpy as np
import time

import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import *
from modules import *
from trainer import *

env = HarlowEnv()
env = MetaLearningWrapper(env)

net = RecurrentActorCriticPolicy(
    feature_dim = env.observation_space.shape[0],
    action_dim = env.action_space.n,
    policy_hidden_dim = 32,
    value_hidden_dim = 32,
    lstm_hidden_dim = 128,
)

a2c = A2C(
    net = net,
    env = env,
    lr = 3e-4,
    gamma = 0.9,
    beta_v = 0.5,
    beta_e = 0.05,
    # lr_schedule = np.linspace(3e-4, 1e-4, num = 30000),
)

data = a2c.learn(num_episodes = 30000, print_frequency = 1000)

plt.figure()
plt.plot(np.array(data['episode_reward']).reshape(200, -1).mean(axis = 1))
plt.show()