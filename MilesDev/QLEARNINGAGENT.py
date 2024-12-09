import pokerenv.obs_indices as indices
from pokerenv.table import Table
from treys import Deck, Evaluator, Card
from pokerenv.common import GameState, PlayerState, PlayerAction, TablePosition, Action, action_list
from pokerenv.player import Player
from pokerenv.utils import pretty_print_hand, approx_gt, approx_lte
import numpy as np


class QLearningAgent:
    def __init__(self, eta=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = {}  # Q-table to store state-action values
        self.eta = eta      # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        self.actions = []
        self.observations = []
        self.rewards = []

    def get_action(self, observation):
        # Convert observation into a hashable state
        state = tuple(observation)

        # Get valid actions dynamically
        valid_actions = np.argwhere(observation[indices.VALID_ACTIONS] == 1).flatten()

        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in valid_actions}

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            chosen_action = np.random.choice(valid_actions)
        else:
            # Choose action with the highest Q-value
            chosen_action = max(self.q_table[state], key=self.q_table[state].get)

        # Convert chosen action to PlayerAction enum
        chosen_action = PlayerAction(chosen_action)

        # Determine bet size if the action is BET
        bet_size = 0
        if chosen_action == PlayerAction.BET:
            valid_bet_low = observation[indices.VALID_BET_LOW]
            valid_bet_high = observation[indices.VALID_BET_HIGH]
            bet_size = np.random.uniform(valid_bet_low, valid_bet_high)

        # Create the action object
        table_action = Action(chosen_action, bet_size)
        self.actions.append(table_action)
        self.observations.append(observation)
        return table_action

    def _get_reward(self, observation, winners):
        winning_agents = [agent for agent, player in winners]
        winning_players = [player for agent, player in winners]
        pot = observation[indices.POT_SIZE]
        
        if self in winning_agents:
            winning_idx = winning_agents.index(self)
            player = winning_players[winning_idx]
            stack = player.stack
            return 1
        else:
            return 0
    
    def update_policy(self, winners):
        # Update Q-values using the Q-learning update rule
        for i in range(len(self.observations)):
            state = tuple(self.observations[i])
            action = self.actions[i]
            reward = self._get_reward(self.observations[i], winners)
            next_state = tuple(self.observations[i + 1]) if i + 1 < len(self.observations) else None

            if next_state is not None:
                next_max = max(self.q_table[next_state].values())
            else:
                next_max = 0

            self.q_table[state][action.action_type.value] += self.eta * (
                reward + self.gamma * next_max - self.q_table[state][action.action_type.value]
            )

    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []