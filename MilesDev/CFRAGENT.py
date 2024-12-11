import numpy as np
from collections import defaultdict
from pokerenv.common import PlayerAction, Action
import pokerenv.obs_indices as indices

class CFRAgent:
    def __init__(self):
        self.regret_sum = defaultdict(lambda: np.zeros(4))  # 4 possible actions
        self.strategy_sum = defaultdict(lambda: np.zeros(4))
        self.actions = []
        self.observations = []
        self.rewards = []
        
    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        
    def _get_info_set(self, observation):
        """Create a key for the current information set"""
        position = int(observation[indices.ACTING_PLAYER_POSITION])
        pot = observation[indices.POT_SIZE]
        street = observation[15]  # Game street
        
        # Get hole cards
        card1 = (int(observation[8]), int(observation[9]))    # (suit, rank)
        card2 = (int(observation[10]), int(observation[11]))  # (suit, rank)
        
        # Get community cards
        community = []
        for i in range(5):
            if observation[16 + i*2] != 0:
                suit = int(observation[16 + i*2])
                rank = int(observation[17 + i*2])
                community.append((suit, rank))
                
        return (position, int(pot/10), street, card1, card2, tuple(community))
        
    def get_strategy(self, info_set):
        """Get current strategy for this information set"""
        regret = self.regret_sum[info_set]
        normalized = np.maximum(regret, 0)
        sum_positive = np.sum(normalized)
        
        if sum_positive > 0:
            return normalized / sum_positive
        return np.ones(4) / 4  # Default to uniform strategy
        
    def _get_valid_actions(self, observation):
        """Get list of valid actions from observation"""
        valid_actions = []
        # Check each action index directly (1 through 4)
        for i in range(4):
            if observation[1 + i] == 1:  # Valid actions start at index 1
                valid_actions.append(i)
        return valid_actions
        
    def get_action(self, observation):
        """Choose action based on current strategy"""
        self.observations.append(observation)
        info_set = self._get_info_set(observation)
        strategy = self.get_strategy(info_set)
        
        # Get valid actions
        valid_actions = self._get_valid_actions(observation)
                
        # Filter strategy to only valid actions
        valid_probs = np.zeros(4)
        for action in valid_actions:
            valid_probs[action] = strategy[action]
            
        # Normalize probabilities
        sum_probs = np.sum(valid_probs)
        if sum_probs > 0:
            valid_probs /= sum_probs
        else:
            # If all probabilities are zero, use uniform distribution over valid actions
            for action in valid_actions:
                valid_probs[action] = 1.0 / len(valid_actions)
        
        # Choose action
        action_type = PlayerAction(np.random.choice(4, p=valid_probs))
        bet_size = 0
        
        if action_type == PlayerAction.BET:
            bet_size = np.random.uniform(
                observation[indices.VALID_BET_LOW],
                observation[indices.VALID_BET_HIGH]
            )
            
        action = Action(action_type, bet_size)
        self.actions.append(action)
        return action
        
    def update_regret(self, info_set, action, reward, other_action_values):
        """Update regret values"""
        for a in range(4):
            if a == action.action_type.value:
                regret = reward - other_action_values[a]
            else:
                regret = other_action_values[a] - reward
            self.regret_sum[info_set][a] += regret
            
    def update_strategy(self, info_set, strategy):
        """Update strategy sum for averaging"""
        self.strategy_sum[info_set] += strategy