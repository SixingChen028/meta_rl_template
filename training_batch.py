import numpy as np
import time
import pickle
import argparse

import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import *
from modules import *
from trainer_batch import *

if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    # parser.add_argument('--jobid', type = str, help = 'job id')
    # parser.add_argument('--path', type = str, help = 'path to store results')

    # nework parameters
    parser.add_argument('--hidden_size', type = int, default = 128, help = 'lstm hidden size')
    parser.add_argument('--policy_hidden_size', type = int, default = 32, help = 'policy head hidden size')
    parser.add_argument('--value_hidden_size', type = int, default = 32, help = 'value head hidden size')

    # environment parameters
    parser.add_argument('--flip_prob', type = float, default = 0.2, help = 'flip probability')

    # training parameters
    parser.add_argument('--num_episodes', type = int, default = 30000, help = 'training episodes')
    parser.add_argument('--lr', type = float, default = 3e-4, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'batch_size')
    parser.add_argument('--gamma', type = float, default = 0.9, help = 'temporal discount')
    parser.add_argument('--lamda', type = float, default = 1.0, help = 'generalized advantage estimation coefficient')
    parser.add_argument('--beta_v', type = float, default = 0.5, help = 'value loss coefficient')
    parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
    parser.add_argument('--max_grad_norm', type = float, default = 1.0, help = 'gradient clipping')

    args = parser.parse_args()


    env = gym.vector.AsyncVectorEnv([
        lambda: MetaLearningWrapper(
            HarlowEnv(
                flip_prob = args.flip_prob,
            )
        )
        for _ in range(args.batch_size)
    ])

    net = RecurrentActorCriticPolicy(
        feature_dim = env.single_observation_space.shape[0],
        action_dim = env.single_action_space.n,
        lstm_hidden_dim = args.hidden_size,
        policy_hidden_dim = args.policy_hidden_size,
        value_hidden_dim = args.value_hidden_size,
    )

    a2c = BatchMaskA2C(
        net = net,
        env = env,
        lr = args.lr,
        batch_size = args.batch_size,
        gamma = args.gamma,
        lamda = args.lamda,
        beta_v = args.beta_v,
        beta_e =args.beta_e,
        max_grad_norm = args.max_grad_norm,
    )

    data = a2c.learn(num_episodes = args.num_episodes, print_frequency = 10)

    # net_path = os.path.join(args.path, f"net_{args.jobid}.pth")
    # data_path = os.path.join(args.path, f"data_{args.jobid}.p")

    plt.figure()
    plt.plot(np.array(data['episode_reward']).reshape(200, -1).mean(axis = 1))
    plt.show()