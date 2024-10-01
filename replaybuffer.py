import numpy as np
import torch


class BaseReplayBuffer:
    """
    A Base Replay Buffer class.
    """

    def __init__(self):
        """
        Initialize the buffer.
        """
        self.reset()


    def reset(self):
        """
        Reset the rollout.
        """
        pass


    def push(self, **kwargs):
        """
        Push data.
        """
        for key, value in kwargs.items():
            if key in self.rollout:
                self.rollout[key].append(value)
            else:
                raise KeyError(f"Key {key} not found in replay buffer.")
    

    def pull(self, *keys):
        """
        Pull data according to keys.
        """
        return [self.rollout[key] for key in keys]
    

    def reformat(self):
        """
        Reformat rollout data.
        """
        pass


class BatchReplayBuffer(BaseReplayBuffer):
    """
    A Replay Buffer.
    """

    def __init__(self):
        """
        Initialize the buffer.
        """
        super().__init__()


    def reset(self):
        """
        Reset the rollout.
        """
        self.rollout = {
            'masks': [],
            'log_probs': [],
            'entropies': [],
            'values': [],
            'rewards': [],
        }


    def reformat(self):
        """
        Reformat rollout data.
        """
        self.rollout['masks'] = torch.stack(self.rollout['masks'], dim = -1) # (batch_size, seq_len)

        self.rollout['log_probs'] = torch.stack(self.rollout['log_probs'], dim = -1) # (batch_size, seq_len)
        self.rollout['log_probs'] *= self.rollout['masks']

        self.rollout['entropies'] = torch.stack(self.rollout['entropies'], dim = -1) # (batch_size, seq_len)
        self.rollout['entropies'] *= self.rollout['masks']

        self.rollout['values'] = torch.stack(self.rollout['values'], dim = -1) # (batch_size, seq_len + 1)
        self.rollout['values'][:, :-1] *= self.rollout['masks'] # make sure the last column of values is 0 so no need for masking

        self.rollout['rewards'] = torch.stack(self.rollout['rewards'], dim = -1) # (batch_size, seq_len)
        self.rollout['rewards'] *= self.rollout['masks']


if __name__ == '__main__':
    # testing

    buffer = BatchReplayBuffer()
    print(buffer.rollout)

    buffer.push(log_probs = 1, entropies = 2, values = 3, rewards = 4)
    print(buffer.pull('log_probs', 'entropies', 'values', 'rewards'))