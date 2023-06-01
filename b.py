
''' An example of playing randomly in RLCard
'''
import argparse
import pprint
import time
import rlcard
from rlcard.agents.my_agent import MyAgent as MyAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
from rlcard.utils import print_card, reorganize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


env = rlcard.make(
	'no-limit-holdem',
	config={
		'seed': 42,
		'game_num_players':3
	}
)
print(env.num_players)

set_seed(42)

my_agent = MyAgent(
	num_actions = env.num_actions,
	num_obs = 54
	)

random_agent = RandomAgent(num_actions=env.num_actions)
random_agent2 = RandomAgent(num_actions=env.num_actions)

env.set_agents([my_agent, random_agent,random_agent2])

x_axis = []
y_axis = []
y_pretrained_axis = []
avg_payoff, avg_pretrained_payoff = 0, 0

n_episode = 2000000
n_logging = 50
logging_steps = n_episode // n_logging

my_agent.set_logger(tqdm(range(n_episode)))

for episode in my_agent.logger:
	# print(">> Start a new game")

	trajectories, payoffs = env.run(is_training=True)
	tr = reorganize(trajectories, payoffs)
	for ts in tr[0]:
		state, action, reward, next_state, done = tuple(ts)
		my_agent.feed_buffer(state['obs'], action, reward, next_state['obs'], next_state['legal_actions'], done)

	avg_payoff += payoffs[0]
	avg_pretrained_payoff += payoffs[1]
	if episode % logging_steps == 0:
		x_axis.append(episode + 1)
		y_axis.append(avg_payoff / logging_steps)
		y_pretrained_axis.append(avg_pretrained_payoff / logging_steps)
		avg_payoff = 0
		avg_pretrained_payoff = 0
	my_agent.save_checkpoint('./b_ckpt')

plot_steps = my_agent.train_done // n_logging
# sns.lineplot(x=[i for i in range(0, my_agent.train_done, plot_steps)], y=[sum(my_agent.loss_values[i: min(my_agent.train_done, i + plot_steps)]) / (min(my_agent.train_done, i + plot_steps) - i) for i in range(0, my_agent.train_done, plot_steps)], legend='brief', label='loss')

sns.lineplot(x=[i for i in range(0, my_agent.train_done, plot_steps)], y=[sum(my_agent.reward_values[i: min(my_agent.train_done, i + plot_steps)]) / (min(my_agent.train_done, i + plot_steps) - i) for i in range(0, my_agent.train_done, plot_steps)], legend='brief', label='reward')
plt.xlabel('episode')
plt.ylabel('avg reward')
plt.savefig('reward.png')
