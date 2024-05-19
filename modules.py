import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical


class FlattenExtractor(nn.Module):
    """
    A flatten feature extractor.
    """
    def forward(self, x):
        # keep the first dimension while flatten other dimensions
        return x.view(x.size(0), -1)


class MlpExtractor(nn.Module):
    """
    A MLP feture extractor.
    """

    def __init__(self, input_dim, output_dim):
        super(MlpExtractor, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.decoder(x)
        return features


class ValueNet(nn.Module):
    """
    Value baseline network.
    """
    
    def __init__(self, input_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.value_extractor = MlpExtractor(input_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.value_extractor(x)
        value = self.fc_value(features) # ([1, 1])

        return value


class ActionNet(nn.Module):
    """
    Action network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActionNet, self).__init__()
        self.action_extractor = MlpExtractor(input_dim, hidden_dim)
        self.fc_action = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        features = self.action_extractor(x)
        logits = self.fc_action(features)

        policy = F.softmax(logits, dim = -1) # ([1, output_dim])
        dist = Categorical(policy)
        action = dist.sample() # ([1])
        log_prob = dist.log_prob(action) # ([1])
        entropy = dist.entropy() # ([1])
        
        return action, policy, log_prob, entropy


class RecurrentActorCriticPolicy(nn.Module):
    """
    Recurrent actor-critic policy.
    """

    def __init__(
            self,
            feature_dim,
            action_dim,
            policy_hidden_dim = 64,
            value_hidden_dim = 64,
            lstm_hidden_dim = 256
        ):
        super(RecurrentActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.policy_hidden_dim = policy_hidden_dim
        self.value_hidden_dim = value_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.lstm_actor = nn.LSTMCell(feature_dim, lstm_hidden_dim)
        self.lstm_critic = nn.LSTMCell(feature_dim, lstm_hidden_dim)

        # policy and value net
        self.policy_net = ActionNet(lstm_hidden_dim, policy_hidden_dim, action_dim)
        self.value_net = ValueNet(lstm_hidden_dim, value_hidden_dim)


    def forward(self, obs, states_actor = None, states_critic = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # initialize hidden states and cells
        if states_actor is None:
            states_actor = [torch.zeros(features.size(0), self.lstm_actor.hidden_size, device = obs.device) for _ in range(2)]
        if states_critic is None:
            states_critic = [torch.zeros(features.size(0), self.lstm_critic.hidden_size, device = obs.device) for _ in range(2)]

        # iterate one step
        hidden_actor, cell_actor = self.lstm_actor(features, (states_actor[0], states_actor[1]))
        hidden_critic, cell_critic = self.lstm_critic(features, (states_critic[0], states_critic[1]))

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden_actor)

        # compute value
        value = self.value_net(hidden_critic)

        return action, policy, log_prob, entropy, value, (hidden_actor, cell_actor), (hidden_critic, cell_critic)


if __name__ == '__main__':
    policy_net = RecurrentActorCriticPolicy(feature_dim = 60, action_dim = 2)

    # Generate random test input
    test_input = torch.randn((1, 60))

    # Forward pass through the network
    action, policy, log_prob, entropy, value, _, _ = policy_net(test_input)

    print('action:', action)
    print('policy:', policy)
    print('log prob:', log_prob)
    print('entropy:', entropy)
    print('value:', value)
