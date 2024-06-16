import numpy as np
import time

import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F

from replaybuffer import *


class BatchMaskA2C:
    """
    An A2C trainer.
    """

    def __init__(
            self,
            net,
            env,
            lr,
            batch_size,
            gamma,
            lamda,
            beta_v,
            beta_e,
            lr_schedule = None,
            entropy_schedule = None,
            max_grad_norm = 1.,
        ):
        """
        Initialize the trainer.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = net.to(self.device)
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.lamda = lamda
        self.beta_v = beta_v
        self.beta_e = beta_e
        self.lr_schedule = lr_schedule
        self.entropy_schedule = entropy_schedule
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(net.parameters(), lr = self.lr)
    

    def update_model(self, buffer):
        """
        Update model parameters.

        Args:
            buffer: a ReplayBuffer object. rollout include:
                masks: a torch.Tensor with shape (batch_size, seq_len).
                    track ongoing batches. 1 for ongoing time steps and 0 for padding time steps.
                rewards: a torch.Tensor with shape (batch_size, seq_len).
                values: a torch.Tensor with shape (batch_size, seq_len + 1).
                log_probs: a torch.Tensor with shape (batch_size, seq_len).
                entropies: a torch.Tensor with shape (batch_size, seq_len).

        Returns:
            losses_episode: a dictionary. losses for the episode.
        """

        # pull data from buffer
        masks, rewards, values, log_probs, entropies = buffer.pull('masks', 'rewards', 'values', 'log_probs', 'entropies')

        # move data to device
        masks = masks.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        log_probs = log_probs.to(self.device)
        entropies = entropies.to(self.device)

        # compute returns and advantages
        returns, advantages = self.get_discounted_returns(rewards, values) # (batch_size, seq_len)

        # compute policy loss
        policy_loss = -(log_probs * advantages.detach() * masks).sum(axis = 1).mean(axis = 0) # (1,)

        # compute value loss
        value_loss = (F.mse_loss(values[:, :-1], returns, reduction = 'none') * masks).sum(axis = 1).mean(axis = 0) # (1,)

        # compute entropy loss
        entropy_loss = -(entropies * masks).sum(axis = 1).mean(axis = 0) # (1,)

        # compute loss
        loss = (
            policy_loss +
            self.beta_v * value_loss +
            self.beta_e * entropy_loss
        )

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # wrap losses for the episode
        losses_episode = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }

        return losses_episode
    

    def train_one_episode(self):
        """
        Train one episode.

        Returns:
            data_episode: a dictionary. training data for the episode.
        """

        # initialize replay buffer
        buffer = BatchReplayBuffer()
               
        # initialize a trial
        dones = np.zeros(self.batch_size, dtype = bool) # no reset once turned to 1
        mask = torch.ones(self.batch_size).to(self.device)
        states_lstm = None

        # reset environment
        obs, info = self.env.reset()
        obs = torch.Tensor(obs).to(self.device) # (batch_size, feature_dim)

        # iterate through a trial
        while not all(dones):
            # step the net
            action, policy, log_prob, entropy, value, states_lstm = self.net(
                obs, states_lstm
            )
            value = value.view(-1) # (batch_size,)

            # step the env
            obs, reward, done, truncated, info = self.env.step(action)
            obs = torch.Tensor(obs).to(self.device) # (batch_size, feature_dim)
            reward = torch.Tensor(reward).to(self.device) # (batch_size,)

            # push results (make sure shapes are (batch_size,))
            buffer.push(
                masks = mask,
                log_probs = log_prob,
                entropies = entropy,
                values = value,
                rewards = reward,
            )

            # update mask and dones
            # note: the order of the following two lines is crucial
            dones = np.logical_or(dones, done)
            mask = (1 - torch.Tensor(dones)).to(self.device) # keep 0 once a batch is done

        # process the last timestep
        value = torch.zeros((self.batch_size,)).to(self.device) # zero padding for the last time step
        buffer.push(values = value) # push value for the last time step

        # reformat rollout data into (batch_size, seq_len) and mask finished time steps
        buffer.reformat()

        # update model
        losses_episode = self.update_model(buffer)

        # compute reward and length of the epiosde
        episode_reward = (buffer.rollout['rewards'] * buffer.rollout['masks']).sum(axis = 1).mean(axis = 0)
        episode_length = buffer.rollout['masks'].sum(axis = 1).mean(axis = 0)

        # wrap training data for the episode
        data_episode = losses_episode.copy()
        data_episode.update({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
        })

        return data_episode


    def learn(self, num_episodes, print_frequency = 2):
        """
        Train the model.

        Args:
            num_episodes: an integer.
            print_frequency: an integer.

        Returns:
            data: a dictionary. training data.
        """

        # initialize recordings
        data = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'episode_length': [],
            'episode_reward': [],
        }

        # compute number of batches
        num_batch = int(num_episodes / self.batch_size)

        # train the model
        start_time = time.time()
        for batch in range(num_batch):
            # train one episode
            data_episode = self.train_one_episode()
        
            # record training data
            for key, item in data.items():
                data[key].append(data_episode[key])

            # print the training process
            self.print_training_process(
                batch_num = batch + 1,
                time_elapsed = time.time() - start_time,
                data = data_episode,
                print_frequency = print_frequency
            )

            # update learning rate
            if self.lr_schedule is not None:
                self.update_learning_rate(batch)

            # update entropy regularization
            if self.entropy_schedule is not None:
                self.update_entropy_coef(batch)
        
        return data
    

    def get_discounted_returns(self, rewards, values):
        """
        Compute discounted reterns and advantages.

        Args:
            rewards: a torch.Tensor with shape (batch_size, seq_len).
            values: a torch.Tensor with shape (batch_size, seq_len + 1).
                note: finished time steps in rewards and values should already be masked.

        Returns:
            returns: a torch.Tensor with shape (batch_size, seq_len).
            advantages: a torch.Tensor with shape (batch_size, seq_len).
        """

        # get sequence length (max sequence length among batches)
        seq_len = rewards.shape[1]

        # initialize recordings
        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)

        # compute returns and advantages from the last timestep
        # note: final R should always be 0, either by masking or zero padding
        R = values[:, -1].to(self.device)
        advantage = torch.zeros(self.batch_size).to(self.device)
        
        for i in reversed(range(seq_len)):
            # get (v, r, v')
            r = rewards[:, i]
            v = values[:, i]
            v_next = values[:, i + 1]

            # compute return for the timestep
            R = r + R * self.gamma
            returns[:, i] = R

            # compute advantage for the timestep
            delta = r + v_next * self.gamma - v
            advantage = delta + advantage * self.gamma * self.lamda
            advantages[:, i] = advantage
            
        return returns, advantages
    

    def update_learning_rate(self, episode):
        """
        Update the learning rate based on the episode number.
        """

        if episode < len(self.lr_schedule):
            self.lr = self.lr_schedule[episode]

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
    

    def update_entropy_coef(self, episode):
        """
        Update the entropy regularization coefficient based on the episode number.
        """

        if episode < len(self.entropy_schedule):
            self.beta_e = self.entropy_schedule[episode]


    def save_net(self, path):
        """
        Save the net.
        """
        
        torch.save(self.net, path)


    def print_training_process(self, batch_num, time_elapsed, data, print_frequency = 2):
        """
        Print the training process.
        """

        if batch_num % print_frequency == 0:
            ep_num = batch_num * self.batch_size
            print("-------------------------------------------")
            print("| rollout/                |               |")
            print(f"|    ep_len_mean          | {data['episode_length']:<13.1f} |")
            print(f"|    ep_rew_mean          | {data['episode_reward']:<13.5f} |")
            print("| time/                   |               |")
            print(f"|    ep_num               | {ep_num:<13} |")
            print(f"|    batch_num            | {batch_num:<13} |")
            print(f"|    time_elapsed         | {time_elapsed:<13.4f} |")
            print("| train/                  |               |")
            print(f"|    learning_rate        | {self.lr:<13.5f} |")
            print(f"|    loss                 | {data['loss']:<13.4f} |")
            print(f"|    policy_loss          | {data['policy_loss']:<13.4f} |")
            print(f"|    value_loss           | {data['value_loss']:<13.4f} |")
            print(f"|    entropy_loss         | {data['entropy_loss']:<13.4f} |")
            print("-------------------------------------------")








if __name__ == '__main__':
    # testing

    from environment import *
    from networks import *
    from a2c import *


    batch_size = 16

    env = gym.vector.AsyncVectorEnv([
        lambda: MetaLearningWrapper(
            HarlowEnv()
        ) for _ in range(batch_size)
    ])

    net = RecurrentActorCriticPolicy(
        feature_dim = env.single_observation_space.shape[0],
        action_dim = env.single_action_space.n,
        lstm_hidden_dim = 128,
        policy_hidden_dim = 32,
        value_hidden_dim = 32,
    )

    a2c = BatchMaskA2C(
        net = net,
        env = env,
        lr = 3e-4,
        batch_size = batch_size,
        gamma = 0.9,
        lamda = 1.,
        beta_v = 0.05,
        beta_e = 0.05,
        max_grad_norm = 1.,
    )

    data = a2c.learn(num_episodes = 5000)

    plt.figure()
    plt.plot(np.array(data['episode_reward']).reshape(200, -1).mean(axis = 1))
    plt.show()