import numpy as np
import random
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from collections import namedtuple, deque
from itertools import count

Transition = namedtuple("Transition", ('state', 'action', 'reward', 'next_state', 'legal_actions', 'done'))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen = capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(n_observations, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions)
                )

    def forward(self, x):
        return self.net(x.float())


class MyAgent(object):
    def __init__(self,
            num_actions,
            num_obs,
            capacity=20000,
            num_hidden=128,
            LR = 1e-4,
            EPS_START=0.9,
            EPS_END=0.05,
            EPS_DECAY=20000,
            device=None
            ):
        self.use_raw = False
        self.num_actions = num_actions
        self.replay_buffer = ReplayBuffer(capacity)
        if device==None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(num_obs, num_actions, num_hidden).to(self.device)
        self.target_net = DQN(num_obs, num_actions, num_hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = LR, amsgrad=True)
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done=0

    # def step(self,state):
        # return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info

    def step(self, state):
        sample = random.random()
        eps = self.EPS_END+(self.EPS_START-self.EPS_END)*math.exp(-1*self.steps_done/self.EPS_DECAY)
        self.steps_done+=1
        if sample>eps:
            obs = torch.tensor(state['obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs)
            _, action=q_values.max(1)
            action = int(action.item())
            return action
        else:
            return np.random.choice(list(state['legal_actions'].keys())) 
