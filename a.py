
''' An example of playing randomly in RLCard
'''
import argparse
import pprint

import rlcard
from rlcard.agents.my_agent import MyAgent as MyAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
from rlcard.utils import print_card, reorganize

env = rlcard.make(
        'no-limit-holdem',
        config={
            'seed': 42,
            'game_players_num':3
        }
)
print(env.num_players)

# Seed numpy, torch, random
set_seed(42)

# Set agents
my_agent = MyAgent(
        num_actions = env.num_actions,
        num_obs = 54
        )
random_agent = RandomAgent(num_actions=env.num_actions)
random_agent2 = RandomAgent(num_actions=env.num_actions)
env.set_agents([my_agent, random_agent, random_agent2])

for episode in range(300):
	print(">> Start a new game")

	trajectories, payoffs = env.run(is_training=False)
	tr =reorganize(trajectories, payoffs)
	for ts in tr[0]:
		state, action, reward, next_state, done = tuple(ts)
		my_agent.feed_buffer(state['obs'], action, reward, next_state['obs'], next_state['legal_actions'], done)
	final_state = trajectories[0][-1]
	action_record = final_state['action_record']
	state = final_state['raw_obs']
	_action_list = []
	for i in range(1, len(action_record)+1):
		if action_record[-i][0] == state['current_player']:
			break
		_action_list.insert(0, action_record[-i])
	for pair in _action_list:
		print('>> Player', pair[0], 'chooses', pair[1])

# Let's take a look at what the agent card is
	print('===============     Cards all Players    ===============')
	for hands in env.get_perfect_information()['hand_cards']:
		print_card(hands)

	print('===============     Result     ===============')
	if payoffs[0] > 0:
		print('You win {} chips!'.format(payoffs[0]))
	elif payoffs[0] == 0:
		print('It is a tie.')
	else:
		print('You lose {} chips!'.format(-payoffs[0]))
	print('')
	print(payoffs)

