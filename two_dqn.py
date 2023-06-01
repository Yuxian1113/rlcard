
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
		'game_num_players':2
	}
)
print(env.num_players)

set_seed(42)

my_agent = MyAgent(
	num_actions = env.num_actions,
	num_obs = 54
)

my_agent2 = MyAgent(
	num_actions = env.num_actions,
	num_obs = 54
)

my_agents = [my_agent, my_agent2]

random_agent = RandomAgent(num_actions=env.num_actions)
random_agent2 = RandomAgent(num_actions=env.num_actions)
random_agent3 = RandomAgent(num_actions=env.num_actions)
random_agent4 = RandomAgent(num_actions=env.num_actions)

env.set_agents(my_agents)

x_axis = []
y_dqn_axis = []
y_dqn2_axis = []
avg_dqn_payoff, avg_dqn2_payoff = 0, 0

n_episode = 3000000
n_logging = 100
sync_rate = 100000
logging_steps = n_episode // n_logging

my_agents[0].set_logger(tqdm(range(n_episode)))

for episode in my_agents[0].logger:
	# print(">> Start a new game")

	trajectories, payoffs = env.run(is_training=True)
	tr = reorganize(trajectories, payoffs)

	if (episode + 1) % sync_rate == 0:
		my_agents[1].load_checkpoint('./two_dqn_ckpt')

	for ts in tr[0]:
		state, action, reward, next_state, done = tuple(ts)
		my_agents[0].feed_buffer(state['obs'], action, reward, next_state['obs'], next_state['legal_actions'], done)
	
	avg_dqn_payoff += payoffs[0]
	avg_dqn2_payoff += payoffs[1]
	if episode % logging_steps == 0:
		x_axis.append(episode + 1)
		y_dqn_axis.append(avg_dqn_payoff / logging_steps)
		y_dqn2_axis.append(avg_dqn2_payoff / logging_steps)
		avg_dqn_payoff = 0
		avg_dqn2_payoff = 0
	my_agents[0].save_checkpoint('./two_dqn_ckpt')

agent_idx = 0
plot_steps = my_agents[agent_idx].train_done // n_logging

sns.lineplot(x=[i for i in range(0, my_agents[agent_idx].train_done, plot_steps)], y=[sum(my_agents[agent_idx].reward_values[i: min(my_agents[agent_idx].train_done, i + plot_steps)]) / (min(my_agents[agent_idx].train_done, i + plot_steps) - i) for i in range(0, my_agents[agent_idx].train_done, plot_steps)], legend='brief', label='dqn1')

plt.xlabel('episode')
plt.ylabel('avg reward')
plt.savefig('two_dqn_reward.png')
