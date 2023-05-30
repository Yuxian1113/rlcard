
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
		'game_num_players':3
	}
)


my_agent = MyAgent(
	num_actions = env.num_actions,
	num_obs = 54
	)
agent = RandomAgent(num_actions=env.num_actions)
agents = [my_agent]
for i in range(env.num_players-1):
    agents.append(agent)
env.set_agents(agents)

x_axis = []
y_axis = []
y_rand_axis = []
avg_payoff, avg_rand_payoff = 0, 0

n_episode = 1000
n_logging = 50
logging_steps = n_episode // n_logging

for episode in tqdm(range(n_episode)):
	# print(">> Start a new game")

	trajectories, payoffs = env.run(is_training=True)
	tr = reorganize(trajectories, payoffs)
	for ts in tr[0]:
		state, action, reward, next_state, done = tuple(ts)
		my_agent.feed_buffer(state['obs'], action, reward, next_state['obs'], next_state['legal_actions'], done)
	final_state = trajectories[0][-1]
	action_record = final_state['action_record']
	state = final_state['raw_obs']
	_action_list = []
'''        
	for i in range(1, len(action_record)+1):
		if action_record[-i][0] == state['current_player']:
			break
		_action_list.insert(0, action_record[-i])
	for pair in _action_list:
		print('>> Player', pair[0], 'chooses', pair[1])

# Let's take a look at what the agent card is
	print('===============	   Cards all Players	===============')
	for hands in env.get_perfect_information()['hand_cards']:
		print_card(hands)

	print('===============	   Result	  ===============')
	if payoffs[0] > 0:
		print('You win {} chips!'.format(payoffs[0]))
	elif payoffs[0] == 0:
		print('It is a tie.')
	else:
		print('You lose {} chips!'.format(-payoffs[0]))
	print('')
	avg_payoff += payoffs[0]
	avg_rand_payoff += payoffs[1]
	if episode % logging_steps == 0:
		x_axis.append(episode + 1)
		y_axis.append(avg_payoff / logging_steps)
		y_rand_axis.append(avg_rand_payoff / logging_steps)
		avg_payoff = 0
		avg_rand_payoff = 0
	my_agent.save_checkpoint('./ckpt')
'''
sns.lineplot(x=x_axis, y=y_axis, legend='brief', label='DQN')
sns.lineplot(x=x_axis, y=y_rand_axis, legend='brief', label='Random')

plt.xlabel('episode')
plt.ylabel('payoff')
plt.savefig('payoff3.png')
