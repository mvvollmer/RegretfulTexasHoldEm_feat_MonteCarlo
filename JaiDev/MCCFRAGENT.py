import numpy as np
from collections import defaultdict
import math
from pokerenv.common import PlayerAction, Action
import pokerenv.obs_indices as indices

class MCCFRNode:
    def __init__(self, state=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.cumulative_regret = np.zeros(4)  # One for each action
        self.cumulative_strategy = np.zeros(4)
        self.visits = 0
        self.value = 0.0
        
class MCCFRAgent:
    def __init__(self, epsilon=0.05, tau=1000, beta=1e6):
        # Strategy parameters from paper
        self.epsilon = epsilon  # Exploration parameter
        self.tau = tau        # Threshold parameter
        self.beta = beta      # Bonus parameter
        
        # Agent state
        self.nodes = {}
        self.actions = []
        self.observations = []
        self.rewards = []
        
        # Performance tracking
        self.cumulative_reward = 0
        self.hand_count = 0

    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        if len(self.nodes) > 10000:  # Memory management
            self.nodes = {}

    def _get_info_set(self, observation):
        """Create information set key following paper's approach"""
        position = int(observation[indices.ACTING_PLAYER_POSITION])
        pot = observation[indices.POT_SIZE]
        street = observation[15] if len(observation) > 15 else 0
        
        # Get hole cards if available
        cards = tuple()
        if observation[8] != 0:
            cards = (
                (int(observation[8]), int(observation[9])),    # Card 1
                (int(observation[10]), int(observation[11]))   # Card 2
            )
            
        # Get community cards
        community = []
        for i in range(5):
            if len(observation) > 16 + i*2 and observation[16 + i*2] != 0:
                suit = int(observation[16 + i*2])
                rank = int(observation[17 + i*2])
                community.append((suit, rank))
                
        return (position, int(pot/10), street, cards, tuple(community))

    def _get_average_strategy(self, info_set):
        """Get average strategy following paper's approach"""
        if info_set not in self.nodes:
            return np.ones(4) / 4  # Uniform strategy for new states
            
        node = self.nodes[info_set]
        total = np.sum(node.cumulative_strategy)
        
        if total > 0:
            return node.cumulative_strategy / total
        return np.ones(4) / 4

    def _get_current_strategy(self, info_set):
        """Compute current strategy using regret matching from paper"""
        if info_set not in self.nodes:
            self.nodes[info_set] = MCCFRNode()
            
        node = self.nodes[info_set]
        regret_sum = node.cumulative_regret
        
        # Regret matching formula from paper
        positive_regret = np.maximum(regret_sum, 0)
        sum_positive = np.sum(positive_regret)
        
        if sum_positive > 0:
            strategy = positive_regret / sum_positive
        else:
            strategy = np.ones(4) / 4
            
        # Add exploration following paper's ε parameter
        strategy = (1 - self.epsilon) * strategy + self.epsilon * (np.ones(4) / 4)
        
        return strategy

    def _sample_action(self, strategy, valid_actions):
        """Sample action according to strategy but only from valid actions"""
        valid_probs = np.zeros(4)
        for action_type, _ in valid_actions:
            valid_probs[action_type.value] = strategy[action_type.value]
            
        # Normalize probabilities
        sum_probs = np.sum(valid_probs)
        if sum_probs > 0:
            valid_probs /= sum_probs
        else:
            # Fallback to uniform over valid actions
            for action_type, _ in valid_actions:
                valid_probs[action_type.value] = 1.0 / len(valid_actions)
                
        return np.random.choice(4, p=valid_probs)

    def _get_valid_actions(self, observation):
        """Get list of valid actions and their bet ranges"""
        valid_actions = []
        for i in range(4):
            if observation[1 + i] == 1:  # Valid actions start at index 1
                action = PlayerAction(i)
                if action == PlayerAction.BET:
                    bet_range = (observation[indices.VALID_BET_LOW],
                               observation[indices.VALID_BET_HIGH])
                else:
                    bet_range = (0, 0)
                valid_actions.append((action, bet_range))
        return valid_actions

    def get_action(self, observation):
        """Select action using MCCFR as described in the paper"""
        self.observations.append(observation)
        info_set = self._get_info_set(observation)
        valid_actions = self._get_valid_actions(observation)
        
        # Get strategy
        strategy = self._get_current_strategy(info_set)
        action_type = PlayerAction(self._sample_action(strategy, valid_actions))
        
        # Calculate bet size for BET actions using cumulative strategy
        bet_size = 0
        if action_type == PlayerAction.BET:
            bet_range = [va[1] for va in valid_actions if va[0] == PlayerAction.BET][0]
            pot_size = observation[indices.POT_SIZE]
            
            # Scale bet based on pot size and strategy confidence
            if pot_size > 0:
                # Use average strategy confidence to scale bet
                avg_strategy = self._get_average_strategy(info_set)
                confidence = avg_strategy[action_type.value]
                scale = 0.5 + 0.5 * confidence  # Bet between 50-100% of pot based on confidence
                bet_size = min(bet_range[1], max(bet_range[0], pot_size * scale))
            else:
                bet_size = bet_range[0]
                
        action = Action(action_type, bet_size)
        self.actions.append(action)
        return action

    def _update_regret(self, info_set, action, reward, action_values):
        """Update regret values as per paper's CFR formula"""
        if info_set not in self.nodes:
            self.nodes[info_set] = MCCFRNode()
            
        node = self.nodes[info_set]
        
        # Calculate instantaneous regret
        for a in range(4):
            if a == action.action_type.value:
                regret = reward - action_values[a]
            else:
                regret = action_values[a] - reward
            
            # Update cumulative regret with threshold τ
            if abs(regret) > self.tau:
                node.cumulative_regret[a] += regret

    def _update_average_strategy(self, info_set, strategy):
        """Update average strategy contribution"""
        if info_set not in self.nodes:
            self.nodes[info_set] = MCCFRNode()
            
        node = self.nodes[info_set]
        node.visits += 1
        
        # Add current strategy to cumulative strategy with bonus β
        contribution = strategy * (1 + self.beta / (node.visits + self.beta))
        node.cumulative_strategy += contribution

    def update_policy(self, winners):
        """Update regrets and strategies based on game outcome"""
        if not self.observations or not self.actions:
            return

        try:
            # Calculate final reward
            final_observation = self.observations[-1]
            final_reward = self._get_reward(final_observation, winners)
            
            # Track performance
            self.cumulative_reward += final_reward
            self.hand_count += 1
            
            # Update regrets and strategies for all info sets visited
            for obs, action in zip(self.observations, self.actions):
                info_set = self._get_info_set(obs)
                strategy = self._get_current_strategy(info_set)
                
                # Compute counterfactual values
                action_values = np.zeros(4)
                valid_actions = self._get_valid_actions(obs)
                for valid_action, _ in valid_actions:
                    action_values[valid_action.value] = final_reward
                
                # Update regrets and average strategy
                self._update_regret(info_set, action, final_reward, action_values)
                self._update_average_strategy(info_set, strategy)
                
        except Exception as e:
            print(f"Error in MCCFR update_policy: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.reset()

    def _get_reward(self, observation, winners):
        """Calculate reward with proper normalization as per paper"""
        winning_agents = [agent for agent, player in winners]
        winning_players = [player for agent, player in winners]
        pot_size = observation[indices.POT_SIZE]
        
        if self in winning_agents:
            winning_idx = winning_agents.index(self)
            stack = winning_players[winning_idx].stack
            # Normalize by pot size as suggested in paper
            return (stack - observation[indices.ACTING_PLAYER_STACK_SIZE]) / max(pot_size, 1)
        return -1.0