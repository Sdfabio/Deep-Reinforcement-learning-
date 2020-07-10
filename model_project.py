import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64,duel=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first common hidden layer
            fc2_units (int): Number of nodes in second common hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        if duel ==1:
            self.fc4_advantage = nn.Linear(fc2_units,action_size) # For Dueling DQN, stream for Advantage Values
            self.fc4_value = nn.Linear(fc2_units,1) # For Dueling DQN, stream for state values
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if duel ==1:
            x_advantage = self.fc4_advantage(x)
            x_advantage = x_advantage - x_advantage.mean(dim=1, keepdim=True) # To make the sum identifiable [ref: https://arxiv.org/pdf/1511.06581.pdf]
            x_value = self.fc4_value(x)
            out = x_value + x_advantage
        else:
            out=self.fc3(x)
        return out
