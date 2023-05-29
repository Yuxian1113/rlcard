import numpy as np
import os
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
	
	def save_checkpoint(self, model_dir: str):
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		torch.save(self, os.path.join(model_dir, 'dqn.pt'))

	def load_checkpoint(self, model_dir: str):
		if not os.path.exists(os.path.join(model_dir, 'dqn.pt')):
			raise FileNotFoundError(f"The model '{path}' does not exist.")
		checkpoint = torch.load(os.path.join(model_dir, 'dqn.pt'))
		self.load_state_dict(checkpoint.state_dict())

class MyAgent(object):
	def __init__(self,
			num_actions,
			num_obs,
			capacity=20000,
			batch_size=128,
			num_hidden=128,
			train_period=100,
			target_update_period=200,
			GAMMA = 0.99,
			LR = 1e-4,
			EPS_START=0.9,
			EPS_END=0.05,
			EPS_DECAY=20000,
			device=None
			):
		self.use_raw = False
		self.num_actions = num_actions
		self.replay_buffer = ReplayBuffer(capacity)
		self.batch_size=batch_size
		self.train_period = train_period
		self.target_update_period = target_update_period
		if device==None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.gamma = GAMMA
		self.policy_net = DQN(num_obs, num_actions, num_hidden).to(self.device)
		self.target_net = DQN(num_obs, num_actions, num_hidden).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = LR, amsgrad=True)
		self.EPS_START = EPS_START
		self.EPS_END = EPS_END
		self.EPS_DECAY = EPS_DECAY
		self.steps_done=0
		self.train_done=0
	
	def predict(self, state):
		obs = torch.tensor(state['obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
		return self.policy_net(obs).view(-1)


	def eval_step(self, state):
		probs = [0 for _ in range(self.num_actions)]
		q_values = self.predict(state)
		for i in state['legal_actions']:
			probs[i] = q_values[i]
		info = {}
		info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}
		return self.step(state), info

# predict q values of current state and determine the best action 
	def step(self, state):
		sample = random.random()
		eps = self.EPS_END+(self.EPS_START-self.EPS_END)*math.exp(-1*self.steps_done/self.EPS_DECAY)
		if sample>eps:
			q_values = self.predict(state)
			_, action=q_values.max(0)
			action = int(action.item())
			return action
		else:
			return np.random.choice(list(state['legal_actions'].keys())) 
	
	def feed_buffer(self, state, action, reward, next_state, legal_actions, done):
		self.steps_done+=1
		self.replay_buffer.push(state, action, reward, next_state, legal_actions, done)
		if self.steps_done>self.batch_size and self.steps_done%self.train_period==0:
			self.train()

	def train(self):
		self.train_done+=1
		transitions = self.replay_buffer.sample(self.batch_size)
		batch = Transition(*zip(*transitions))
		state_batch=torch.tensor(batch.state, device=self.device)
		action_batch=torch.tensor(batch.action, device=self.device)
		reward_batch=torch.tensor(batch.reward, device=self.device)
		# computes the q_values of all actions under the state and then choose the one corresponds to the actions selected
		Vt = self.policy_net(state_batch).gather(1, torch.tensor(batch.action, device=self.device).unsqueeze(1)).view(-1) 
		Vt_next = self.target_net(state_batch).max(1)[0] 
		E_Vt = (Vt_next*self.gamma)+reward_batch

		criterion = nn.SmoothL1Loss()
		loss = criterion(Vt, E_Vt.unsqueeze(1))
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
		self.optimizer.step()
		if self.train_done>0 and self.train_done%self.target_update_period==0:
			self.target_net.load_state_dict(self.policy_net.state_dict())

	def save_checkpoint(self, model_dir: str):
		self.policy_net.save_checkpoint(model_dir)
	
	def load_checkpoint(self, model_dir: str):
		self.policy_net.load_checkpoint(model_dir)
