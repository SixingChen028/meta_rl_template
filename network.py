import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    """
    A torch Categorical class with action masking.
    """

    def __init__(self, logits, mask):
        self.mask = mask

        if mask is None:
            super(CategoricalMasked, self).__init__(logits = logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype = logits.dtype
            )
            logits = torch.where(self.mask, logits, self.mask_value)
            super(CategoricalMasked, self).__init__(logits = logits)


    def entropy(self):
        if self.mask is None:
            return super().entropy()
        
        p_log_p = self.logits * self.probs

        # compute entropy with possible actions only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype = p_log_p.dtype, device = p_log_p.device),
        )

        return -torch.sum(p_log_p, axis = 1)
    

class FlattenExtractor(nn.Module):
    """
    A flatten feature extractor.
    """
    def forward(self, x):
        # keep the first dimension while flatten other dimensions
        return x.view(x.size(0), -1)


class ValueNet(nn.Module):
    """
    Value baseline network.
    """
    
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.fc_value = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        value = self.fc_value(x) # (batch_size, 1)

        return value


class ActionNet(nn.Module):
    """
    Action network.
    """

    def __init__(self, input_dim, output_dim):
        super(ActionNet, self).__init__()
        self.fc_action = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, mask = None):
        self.logits = self.fc_action(x) # record logits for later analyses

        # no action masking
        if mask == None:
            dist = Categorical(logits = self.logits)
        
        # with action masking
        elif mask != None:
            dist = CategoricalMasked(logits = self.logits, mask = mask)
        
        policy = dist.probs # (batch_size, output_dim)
        action = dist.sample() # (batch_size,)
        log_prob = dist.log_prob(action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)
        
        return action, policy, log_prob, entropy


class LSTMRecurrentActorCriticPolicy(nn.Module):
    """
    LSTM recurrent actor-critic policy.
    """

    def __init__(
            self,
            feature_dim,
            action_dim,
            lstm_hidden_dim = 128,
        ):
        super(LSTMRecurrentActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.lstm_actor = nn.LSTMCell(feature_dim, lstm_hidden_dim)
        self.lstm_critic = nn.LSTMCell(feature_dim, lstm_hidden_dim)

        # policy and value net
        self.policy_net = ActionNet(lstm_hidden_dim, action_dim)
        self.value_net = ValueNet(lstm_hidden_dim)


    def forward(self, obs, states_lstm = None, mask = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # initialize hidden states and cells
        if states_lstm is None:
            states_actor = [torch.zeros(features.size(0), self.lstm_actor.hidden_size, device = obs.device) for _ in range(2)]
            states_critic = [torch.zeros(features.size(0), self.lstm_critic.hidden_size, device = obs.device) for _ in range(2)]
        else:
            states_actor, states_critic = states_lstm

        # iterate one step
        hidden_actor, cell_actor = self.lstm_actor(features, (states_actor[0], states_actor[1]))
        hidden_critic, cell_critic = self.lstm_critic(features, (states_critic[0], states_critic[1]))

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden_actor, mask)

        # compute value
        value = self.value_net(hidden_critic)

        return action, policy, log_prob, entropy, value, [(hidden_actor, cell_actor), (hidden_critic, cell_critic)]


class SharedLSTMRecurrentActorCriticPolicy(nn.Module):
    """
    LSTM recurrent actor-critic policy with shared actor and critic.
    """

    def __init__(
            self,
            feature_dim,
            action_dim,
            lstm_hidden_dim = 128,
        ):
        super(SharedLSTMRecurrentActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.lstm = nn.LSTMCell(feature_dim, lstm_hidden_dim)

        # policy and value net
        self.policy_net = ActionNet(lstm_hidden_dim, action_dim)
        self.value_net = ValueNet(lstm_hidden_dim)


    def forward(self, obs, states_lstm = None, mask = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # initialize hidden states and cells
        if states_lstm is None:
            states_lstm = [torch.zeros(features.size(0), self.lstm.hidden_size, device = obs.device) for _ in range(2)]
        
        # iterate one step
        hidden, cell = self.lstm(features, (states_lstm[0], states_lstm[1]))

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden, mask)

        # compute value
        value = self.value_net(hidden)

        return action, policy, log_prob, entropy, value, (hidden, cell)


class GRURecurrentActorCriticPolicy(nn.Module):
    """
    GRU recurrent actor-critic policy.
    """

    def __init__(
            self,
            feature_dim,
            action_dim,
            gru_hidden_dim = 128,
        ):
        super(GRURecurrentActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gru_hidden_dim = gru_hidden_dim

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.gru_actor = nn.GRUCell(feature_dim, gru_hidden_dim)
        self.gru_critic = nn.GRUCell(feature_dim, gru_hidden_dim)

        # policy and value net
        self.policy_net = ActionNet(gru_hidden_dim, action_dim)
        self.value_net = ValueNet(gru_hidden_dim)


    def forward(self, obs, states_gru = None, mask = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # initialize hidden states
        if states_gru is None:
            states_actor = torch.zeros(features.size(0), self.gru_actor.hidden_size, device = obs.device)
            states_critic = torch.zeros(features.size(0), self.gru_critic.hidden_size, device = obs.device)
        else:
            states_actor, states_critic = states_gru

        # iterate one step
        hidden_actor = self.gru_actor(features, states_actor)
        hidden_critic = self.gru_critic(features, states_critic)

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden_actor, mask)

        # compute value
        value = self.value_net(hidden_critic)

        return action, policy, log_prob, entropy, value, [hidden_actor, hidden_critic]


class SharedGRURecurrentActorCriticPolicy(nn.Module):
    """
    GRU recurrent actor-critic policy with shared actor and critic.
    """

    def __init__(
            self,
            feature_dim,
            action_dim,
            gru_hidden_dim = 128,
        ):
        super(SharedGRURecurrentActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gru_hidden_dim = gru_hidden_dim

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.gru = nn.GRUCell(feature_dim, gru_hidden_dim)

        # policy and value net
        self.policy_net = ActionNet(gru_hidden_dim, action_dim)
        self.value_net = ValueNet(gru_hidden_dim)


    def forward(self, obs, states_gru = None, mask = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # initialize hidden states
        if states_gru is None:
            states_gru = torch.zeros(features.size(0), self.gru.hidden_size, device = obs.device)
        
        # iterate one step
        hidden = self.gru(features, states_gru)

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden, mask)

        # compute value
        value = self.value_net(hidden)

        return action, policy, log_prob, entropy, value, hidden



if __name__ == '__main__':
    # testing

    feature_dim = 60
    action_dim = 3
    batch_size = 16


    net = LSTMRecurrentActorCriticPolicy(
        feature_dim = feature_dim,
        action_dim = action_dim,
    )

    # net = SharedLSTMRecurrentActorCriticPolicy(
    #     feature_dim = feature_dim,
    #     action_dim = action_dim,
    # )

    # net = GRURecurrentActorCriticPolicy(
    #     feature_dim = feature_dim,
    #     action_dim = action_dim,
    # )

    # net = SharedGRURecurrentActorCriticPolicy(
    #     feature_dim = feature_dim,
    #     action_dim = action_dim,
    # )

    # generate random test input
    test_input = torch.randn((batch_size, feature_dim))
    test_mask = torch.randint(0, 2, size = (batch_size, action_dim), dtype = torch.bool)

    # forward pass through the network
    action, policy, log_prob, entropy, value, states_lstm = net(test_input, mask = test_mask)

    print('action:', action)
    print('policy:', policy)
    print('log prob:', log_prob)
    print('entropy:', entropy)
    print('value:', value)
    print('lstm states:', states_lstm)
