import os
import time
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import gymnasium as gym

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

        self.net = net
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
            buffer: a ReplayBuffer object. rollout includes:
                masks: a torch.Tensor with shape (batch_size, seq_len).
                    track ongoing batches. 1 for ongoing time steps and 0 for padding time steps.
                rewards: a torch.Tensor with shape (batch_size, seq_len).
                values: a torch.Tensor with shape (batch_size, seq_len + 1).
                log_probs: a torch.Tensor with shape (batch_size, seq_len).
                entropies: a torch.Tensor with shape (batch_size, seq_len).

        Returns:
            losses_batch: a dictionary. losses for the batch.
        """

        # pull data from buffer
        masks, rewards, values, log_probs, entropies = buffer.pull('masks', 'rewards', 'values', 'log_probs', 'entropies')

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

        # wrap losses for the batch
        losses_batch = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }

        return losses_batch
    

    def train_one_batch(self):
        """
        Train one batch.

        Returns:
            data_batch: a dictionary. training data for the batch.
        """

        # initialize replay buffer
        buffer = BatchReplayBuffer()
               
        # initialize a trial
        dones = np.zeros(self.batch_size, dtype = bool) # no reset once turned to 1
        mask = torch.ones(self.batch_size)
        states = None

        # reset environment
        obs, info = self.env.reset()
        obs = torch.Tensor(obs) # (batch_size, feature_dim)
        action_mask = torch.tensor(np.stack(info['mask'])) # (batch_size, action_dim), bool

        # iterate through a trial
        while not all(dones):
            # step the net
            action, policy, log_prob, entropy, value, states = self.net(
                obs, states, action_mask,
            )
            value = value.view(-1) # (batch_size,)

            # step the env
            obs, reward, done, truncated, info = self.env.step(action)
            obs = torch.Tensor(obs) # (batch_size, feature_dim)
            reward = torch.Tensor(reward) # (batch_size,)
            action_mask = torch.tensor(np.stack(info['mask'])) # (batch_size, action_dim), bool

            # push results (make sure shapes are (batch_size,))
            buffer.push(
                masks = mask, # (batch_size,)
                log_probs = log_prob, # (batch_size,)
                entropies = entropy, # (batch_size,)
                values = value, # (batch_size,)
                rewards = reward, # (batch_size,)
            )

            # update mask and dones
            # note: the order of the following two lines is crucial
            dones = np.logical_or(dones, done)
            mask = (1 - torch.Tensor(dones)) # keep 0 once a batch is done

        # process the last timestep
        value = torch.zeros((self.batch_size,)) # zero padding for the last time step
        buffer.push(values = value) # push value for the last time step

        # reformat rollout data into (batch_size, seq_len) and mask finished time steps
        buffer.reformat()

        # update model
        losses_batch = self.update_model(buffer)

        # compute reward and length of the epiosde
        episode_length = buffer.rollout['masks'].sum(axis = 1).mean(axis = 0)
        episode_reward = (buffer.rollout['rewards'] * buffer.rollout['masks']).sum(axis = 1).mean(axis = 0)

        # wrap training data for the batch
        data_batch = losses_batch.copy()
        data_batch.update({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
        })

        return data_batch


    def learn(
            self,
            num_episode,
            print_frequency = None,
            checkpoint_frequency = None,
            checkpoint_path = None,
        ):
        """
        Train the model.

        Args:
            num_episode: an integer.
            print_frequency: an integer.
            checkpoint_frequency: an integer.
            checkpoint_path: a string.

        Returns:
            data: a dictionary. training data.
        """

        # initialize recordings
        self.data = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'episode_length': [],
            'episode_reward': [],
        }

        # compute number of batches
        num_batches = int(num_episode / self.batch_size)

        # train the model
        start_time = time.time()
        for batch in range(num_batches):
            # train one batch
            data_batch = self.train_one_batch()
        
            # record training data
            for key, item in self.data.items():
                self.data[key].append(data_batch[key])

            # print the training process
            if print_frequency is not None and (batch + 1) % print_frequency == 0:
                self.print_training_process(
                    batch_num = batch + 1,
                    time_elapsed = time.time() - start_time,
                    data = data_batch
                )
            
            # save check point
            if checkpoint_frequency is not None and (batch + 1) % checkpoint_frequency == 0:
                self.save_net(os.path.join(checkpoint_path, f'net_{batch + 1}.pth'))

            # update learning rate
            if self.lr_schedule is not None:
                self.update_learning_rate(batch)

            # update entropy regularization
            if self.entropy_schedule is not None:
                self.update_entropy_coef(batch)
        
        return self.data
    

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
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # compute returns and advantages from the last timestep
        # note: final R should always be 0, either by masking or zero padding
        R = values[:, -1]
        advantage = torch.zeros(self.batch_size)
        
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
    

    def update_learning_rate(self, batch):
        """
        Update the learning rate based on the batch number.
        """

        if batch < len(self.lr_schedule):
            self.lr = self.lr_schedule[batch]

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
    

    def update_entropy_coef(self, batch):
        """
        Update the entropy regularization coefficient based on the batch number.
        """

        if batch < len(self.entropy_schedule):
            self.beta_e = self.entropy_schedule[batch]


    def save_net(self, path):
        """
        Save the net.
        """
        
        torch.save(self.net, path)

    
    def save_data(self, path):
        """
        Save the data given the whole path.
        """
        
        pickle.dump(self.data, open(path, 'wb'))


    def print_training_process(self, batch_num, time_elapsed, data):
        """
        Print the training process.
        """

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


