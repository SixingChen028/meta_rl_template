import numpy as np
import time
import os

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from sb3_contrib import RecurrentPPO


class HarlowEnv(gym.Env):
    """
    A bandit environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, flip_prob = 0.2):
        """
        Construct an environment.
        """

        # max number of trials per episode
        self.num_trials = 20

        # flip probability
        self.flip_prob = flip_prob

        self.action_space = Discrete(3)
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = (1,))


    def reset(self, seed = None, option = {}):
        """
        Reset the environment.
        """

        # reset the environment
        self.num_completed = 0
        self.stage = 'fixation'
        self.correct_answer = np.random.randint(0, 2)

        obs = np.array([1.])
        info = {
            'correct_answer': self.correct_answer,
        }

        return obs, info
    

    def step(self, action):
        """
        Step the environment.
        """

        done = False

        # fixation stage
        if self.stage == 'fixation':
            self.stage = 'decision'

            # fixation action
            if action == 2:
                reward = 0.
            
            # decision action
            else:
                reward = -1.
            
            obs = np.array([0.])
        
        # decision stage
        elif self.stage == 'decision':
            self.stage = 'fixation'
            self.num_completed += 1
            self.flip_bandit()

            if action == self.correct_answer:
                reward = 1.
            else:
                reward = -1.
            
            obs = np.array([1.])
        
        if self.num_completed >= self.num_trials + np.random.randint(0, 10, 1): # test for different episode length
            done = True
        
        info = {
            'correct_answer': self.correct_answer,
        }

        return obs, reward, done, False, info
    

    def flip_bandit(self):
        """
        Flip the bandit.
        """

        if np.random.random() < self.flip_prob:
            self.correct_answer = 1 - self.correct_answer

    
    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot



class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        self.init_prev_variables()

        new_observation_shape = (
            self.env.observation_space.shape[0] +
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        obs_wrapped = self.wrap_obs(obs)

        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        self.init_prev_variables()

        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs,
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward
        ])
        return obs_wrapped



if __name__ == '__main__':
    # testing
    
    env = HarlowEnv()
    env = MetaLearningWrapper(env)
    

    # model = RecurrentPPO(
    #     policy = 'MlpLstmPolicy',
    #     env = env,
    #     verbose = 1,
    #     learning_rate = 1e-4,
    #     n_steps = 20,
    #     gamma = 0.9,
    #     ent_coef = 0.05,
    # )

    # model.learn(total_timesteps = 1000000)

    for i in range(50):

        obs, info = env.reset()
        print('initial obs:', obs)
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(
                'obs:', obs, '|',
                'action:', action, '|',
                'correct answer:', info['correct_answer'], '|',
                'reward:', reward, '|',
                'done:', done, '|',
            )
