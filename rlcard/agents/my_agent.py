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
		fc = []
		fc.append(nn.Linear(n_observations, n_hidden, bias=True))
		fc.append(nn.Tanh())
		fc.append(nn.Linear(n_hidden, n_hidden, bias=True))
		fc.append(nn.Tanh())
		fc.append(nn.Linear(n_hidden, n_actions, bias=True))
		self.net = nn.Sequential(*fc)

	def forward(self, x):
		return self.net(x.to(torch.float32))
	
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
				batch_size=32,
				num_hidden=128,
				train_period=1,
				target_update_period=1000,
				GAMMA = 0.99,
				LR = 5e-5,
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
		self.policy_net.eval()
		self.target_net.eval()
		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = LR, amsgrad=True)
		self.EPS_START = EPS_START
		self.EPS_END = EPS_END
		self.EPS_DECAY = EPS_DECAY
		self.steps_done=0
		self.train_done=0
		self.logger = None
		self.loss_values = []
		self.reward_values = []
	
	def set_logger(self, logger):
		self.logger = logger
	
	def predict(self, state):
		with torch.no_grad():
			obs = torch.tensor(state['obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
			q_values = self.policy_net(obs).view(-1).detach().cpu().numpy()
			masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
			legal_actions = list(state['legal_actions'].keys())
			masked_q_values[legal_actions] = q_values[legal_actions]
			return masked_q_values

	def eval_step(self, state):
		q_values = self.predict(state)
		best_action = np.argmax(q_values)
		info = {}
		info['values'] = {state['raw_legal_actions'][i]: q_values[list(state['legal_actions'].keys())[i]].item() for i in range(len(state['legal_actions']))}
		return best_action, info

# predict q values of current state and determine the best action 
	def step(self, state):
		q_values = self.predict(state)
		eps = self.EPS_END+(self.EPS_START-self.EPS_END)*math.exp(-1*self.steps_done/self.EPS_DECAY)
		legal_actions = list(state['legal_actions'].keys())
		probs = np.ones(len(legal_actions), dtype=float) * eps / len(legal_actions)
		best_action_idx = legal_actions.index(np.argmax(q_values))
		probs[best_action_idx] += (1.0 - eps)
		action_idx = np.random.choice(np.arange(len(probs)), p=probs)
		return legal_actions[action_idx]
	
	def feed_buffer(self, state, action, reward, next_state, legal_actions, done):
		self.steps_done+=1
		self.replay_buffer.push(state, action, reward, next_state, legal_actions, done)
		if self.steps_done>self.batch_size * 5 and self.steps_done%self.train_period==0:
			self.train()

	def train(self):
		self.train_done+=1
		transitions = self.replay_buffer.sample(self.batch_size)
		batch = Transition(*zip(*transitions))
		state_batch=torch.tensor(batch.state, device=self.device)
		next_state_batch=torch.tensor(batch.next_state, device=self.device)
		action_batch=torch.tensor(batch.action, device=self.device)
		reward_batch=torch.tensor(batch.reward, device=self.device)
		done_batch=torch.tensor(np.invert(batch.done), device=self.device)
		legal_actions_batch = batch.legal_actions
		avg_reward = sum(batch.reward) / len(batch.reward)
		# computes the q_values of all actions under the state and then choose the one corresponds to the actions selected
		with torch.no_grad():
			Vt_next = self.target_net(next_state_batch)
			q_next_values = self.policy_net(next_state_batch).cpu().detach().numpy()
			mask_q_next_values = -np.inf * np.ones((self.batch_size, self.num_actions), dtype=float)
			best_actions = np.zeros(self.batch_size, dtype=int)
			for b in range(self.batch_size):
				legal_actions = list(legal_actions_batch[b].keys())
				mask_q_next_values[b][legal_actions] = q_next_values[b][legal_actions]
				best_actions[b] = np.argmax(mask_q_next_values[b])
			Vt_next = Vt_next.gather(1, torch.from_numpy(best_actions.copy()).long().to(self.device).unsqueeze(1)).view(-1)
		self.policy_net.train()
		q_value = self.policy_net(state_batch)
		Vt = q_value.gather(1, torch.tensor(batch.action, device=self.device).unsqueeze(1)).view(-1)

		E_Vt = (Vt_next * self.gamma * done_batch.long())+reward_batch

		criterion = nn.MSELoss()
		loss = criterion(Vt, E_Vt)
		if type(self.logger) != type(None):
			self.logger.set_postfix({'loss': loss.item(), 'reward': avg_reward})
			self.loss_values.append(loss.item())
			self.reward_values.append(avg_reward)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.policy_net.eval()
		if self.train_done>0 and self.train_done%self.target_update_period==0:
			self.target_net.load_state_dict(self.policy_net.state_dict())

	def save_checkpoint(self, model_dir: str):
		self.policy_net.save_checkpoint(model_dir)
	
	def load_checkpoint(self, model_dir: str):
		self.policy_net.load_checkpoint(model_dir)
		self.target_net.load_state_dict(self.policy_net.state_dict())
