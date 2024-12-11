import numpy as np
from collections import defaultdict
import math
from pokerenv.common import PlayerAction, Action
import pokerenv.obs_indices as indices

class HybridNode:
    def __init__(self, state=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 1.0
        self.regret_sum = np.zeros(4)  # CFR regret sum
        self.strategy_sum = np.zeros(4)  # CFR strategy sum
        self.untried_actions = None

class MCTSCFRAgent:
    def __init__(self, c_param=1.414, n_simulations=100, eta=0.1, exploration_factor=0.4):
        self.c_param = c_param
        self.n_simulations = n_simulations
        self.eta = eta
        self.exploration_factor = exploration_factor
        self.actions = []
        self.observations = []
        self.rewards = []
        self.nodes = {}
        self.regret_sum = defaultdict(lambda: np.zeros(4))
        self.strategy_sum = defaultdict(lambda: np.zeros(4))
        self.total_reward = 0
        self.hand_count = 0

    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def _get_info_set(self, observation):
        """Create enhanced information set key"""
        position = int(observation[indices.ACTING_PLAYER_POSITION])
        stack = observation[indices.ACTING_PLAYER_STACK_SIZE]
        pot = observation[indices.POT_SIZE]
        street = int(observation[15] if len(observation) > 15 else 0)
        
        # Include hole cards if available
        cards = []
        if observation[8] != 0:
            cards = [
                (int(observation[8]), int(observation[9])),    # Card 1
                (int(observation[10]), int(observation[11]))   # Card 2
            ]
            
        # Include community cards
        community = []
        for i in range(5):
            if len(observation) > 16 + i*2 and observation[16 + i*2] != 0:
                suit = int(observation[16 + i*2])
                rank = int(observation[17 + i*2])
                community.append((suit, rank))
                
        return (position, int(stack/10), int(pot/10), street, tuple(cards), tuple(community))

    def _get_valid_actions(self, observation):
        """Get list of valid actions and their bet ranges"""
        valid_actions = []
        
        # Check valid actions from observation
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

    def get_cfr_strategy(self, info_set):
        """Get current CFR strategy with improved normalization"""
        regret = self.regret_sum[info_set]
        normalized = np.maximum(regret, 0)
        sum_positive = np.sum(normalized)
        
        if sum_positive > 0:
            strategy = normalized / sum_positive
        else:
            strategy = np.ones(4) / 4
            
        # Apply minimal exploration
        strategy = 0.95 * strategy + 0.05 * (np.ones(4) / 4)
        return strategy

    def _select_node(self, node, observation):
        """Select best child node using improved selection criteria"""
        best_score = float('-inf')
        best_action = None
        best_child = None

        info_set = self._get_info_set(observation)
        cfr_strategy = self.get_cfr_strategy(info_set)
        valid_actions = self._get_valid_actions(observation)
        
        # Filter available actions
        available_actions = {action: child for action, child in node.children.items()
                           if any(va[0] == action.action_type for va in valid_actions)}

        for action, child in available_actions.items():
            # Enhanced selection formula
            exploitation = child.value / (child.visits + 1e-8)
            exploration = self.c_param * np.sqrt(math.log(node.visits + 1) / (child.visits + 1e-8))
            cfr_influence = self.exploration_factor * cfr_strategy[action.action_type.value]
            
            # Consider pot odds for betting actions
            if action.action_type == PlayerAction.BET:
                pot_size = observation[indices.POT_SIZE]
                if pot_size > 0:
                    pot_odds = action.bet_amount / (pot_size + action.bet_amount)
                    exploitation *= (1 + pot_odds)
            
            score = exploitation + exploration + cfr_influence
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _simulate(self, observation):
        """Enhanced simulation with pot odds and hand strength consideration"""
        info_set = self._get_info_set(observation)
        strategy = self.get_cfr_strategy(info_set)
        
        sim_reward = 0
        pot_size = observation[indices.POT_SIZE]
        stack_size = observation[indices.ACTING_PLAYER_STACK_SIZE]
        
        # Consider pot odds and stack size
        for action_type, bet_range in self._get_valid_actions(observation):
            action_prob = strategy[action_type.value]
            
            if action_type == PlayerAction.FOLD:
                sim_reward += action_prob * (-0.5)  # Normalized loss
            elif action_type == PlayerAction.CHECK:
                sim_reward += action_prob * 0.1
            elif action_type == PlayerAction.CALL:
                if pot_size > 0:
                    pot_odds = bet_range[0] / (pot_size + bet_range[0])
                    sim_reward += action_prob * (0.2 * (1 - pot_odds))
            elif action_type == PlayerAction.BET:
                if pot_size > 0:
                    max_bet = min(bet_range[1], stack_size)
                    pot_odds = max_bet / (pot_size + max_bet)
                    sim_reward += action_prob * (0.3 * pot_odds)
            
        return sim_reward

    def get_action(self, observation):
        """Get action with improved decision making"""
        self.observations.append(observation)
        info_set = self._get_info_set(observation)
        
        # Initialize node if needed
        if info_set not in self.nodes:
            self.nodes[info_set] = HybridNode(state=info_set)
        root = self.nodes[info_set]
        
        # Run MCTS simulations
        for _ in range(self.n_simulations):
            node = root
            current_obs = observation.copy()
            
            # Selection
            while node.untried_actions is None or not node.untried_actions:
                if not node.children:
                    break
                action, node = self._select_node(node, current_obs)
                if node is None:
                    break
            
            # Expansion and Simulation
            if node is not None:
                action, new_node = self._expand_node(node, current_obs)
                if new_node is not None:
                    reward = self._simulate(current_obs)
                    self._backpropagate(new_node, reward)
        
        # Choose best action
        strategy = self.get_cfr_strategy(info_set)
        valid_actions = self._get_valid_actions(observation)
        
        best_action = None
        best_value = float('-inf')
        
        for action_type, bet_range in valid_actions:
            action_value = 0
            
            # Create action with appropriate bet size
            bet_size = 0
            if action_type == PlayerAction.BET and bet_range[1] > 0:
                # Smart bet sizing based on pot and stack
                pot_size = observation[indices.POT_SIZE]
                stack_size = observation[indices.ACTING_PLAYER_STACK_SIZE]
                
                if pot_size > 0:
                    # Bet between 0.5 and 1 pot size
                    bet_size = min(max(pot_size * 0.5, bet_range[0]), 
                                 min(pot_size, bet_range[1], stack_size))
                else:
                    bet_size = bet_range[0]
                    
            action = Action(action_type, bet_size)
            
            # Calculate action value
            if action in root.children:
                mcts_value = root.children[action].value / (root.children[action].visits + 1e-8)
            else:
                mcts_value = 0
                
            cfr_value = strategy[action_type.value]
            combined_value = (1 - self.exploration_factor) * mcts_value + self.exploration_factor * cfr_value
            
            # Add pot odds consideration
            if action_type == PlayerAction.BET and pot_size > 0:
                pot_odds = bet_size / (pot_size + bet_size)
                combined_value *= (1 + pot_odds)
            
            if combined_value > best_value:
                best_value = combined_value
                best_action = action
        
        if best_action is None:
            # Fallback to reasonable default action
            action_type, bet_range = valid_actions[0]
            best_action = Action(action_type, 0)
        
        self.actions.append(best_action)
        return best_action

    def _get_reward(self, observation, winners):
        """Enhanced reward calculation"""
        winning_agents = [agent for agent, player in winners]
        winning_players = [player for agent, player in winners]
        pot_size = observation[indices.POT_SIZE]
        
        if self in winning_agents:
            winning_idx = winning_agents.index(self)
            player = winning_players[winning_idx]
            
            # Calculate reward based on profit relative to pot size
            reward = (player.stack - observation[indices.ACTING_PLAYER_STACK_SIZE]) / max(pot_size, 1)
            return max(min(reward, 1.0), -1.0)  # Normalize between -1 and 1
        return -1.0

    def update_policy(self, winners):
        """Update both MCTS and CFR components with improved learning"""
        if not self.observations or not self.actions:
            return

        try:
            final_observation = self.observations[-1]
            reward = self._get_reward(final_observation, winners)
            
            # Update statistics
            self.total_reward += reward
            self.hand_count += 1
            
            # Decay learning parameters
            if self.hand_count % 1000 == 0:
                self.exploration_factor *= 0.95
                self.c_param *= 0.95
            
            # Update CFR values
            for obs, action in zip(self.observations, self.actions):
                info_set = self._get_info_set(obs)
                strategy = self.get_cfr_strategy(info_set)
                
                # Calculate regret updates
                action_values = np.zeros(4)
                action_values[action.action_type.value] = reward
                
                # Update regrets with decay
                for a in range(4):
                    regret = action_values[a] - reward
                    self.regret_sum[info_set][a] = self.regret_sum[info_set][a] * 0.95 + regret
                
                # Update strategy sum
                self.strategy_sum[info_set] = self.strategy_sum[info_set] * 0.95 + strategy
            
            # Clear MCTS tree periodically to prevent memory issues
            if len(self.nodes) > 10000:
                self.nodes = {}
                
        except Exception as e:
            print(f"Error in update_policy: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.reset()