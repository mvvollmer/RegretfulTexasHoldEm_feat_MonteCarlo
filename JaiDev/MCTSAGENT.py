import numpy as np
from collections import defaultdict
import math
from pokerenv.common import PlayerAction, Action
import pokerenv.obs_indices as indices

class MCTSNode:
    def __init__(self, state=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.depth = 0 if parent is None else parent.depth + 1

class MCTSAgent:
    def __init__(self, c_param=1.414, n_simulations=50, max_depth=10):
        self.c_param = c_param
        self.n_simulations = n_simulations
        self.max_depth = max_depth  # Add depth limit
        self.actions = []
        self.observations = []
        self.rewards = []
        self.nodes = {}
        self.cache = {}  # Add state value cache

    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        if len(self.nodes) > 10000:  # Prevent memory issues
            self.nodes = {}
            self.cache = {}

    def _get_state_key(self, observation):
        """Create a more compact state representation"""
        position = int(observation[indices.ACTING_PLAYER_POSITION])
        stack = int(observation[indices.ACTING_PLAYER_STACK_SIZE] / 100)  # Reduce granularity
        pot = int(observation[indices.POT_SIZE] / 100)  # Reduce granularity
        
        # Include cards if available
        cards = tuple()
        if observation[8] != 0:  # If cards exist
            cards = (
                (int(observation[8]), int(observation[9])),    # Card 1
                (int(observation[10]), int(observation[11]))   # Card 2
            )
        
        return (position, stack, pot, cards)

    def _get_valid_actions(self, observation):
        """Get list of valid actions and their bet ranges"""
        valid_actions = []
        for i in range(4):  # 4 possible actions
            if observation[1 + i] == 1:
                action = PlayerAction(i)
                bet_range = (0, 0)
                if action == PlayerAction.BET:
                    bet_range = (
                        observation[indices.VALID_BET_LOW],
                        min(observation[indices.VALID_BET_HIGH], observation[indices.ACTING_PLAYER_STACK_SIZE])
                    )
                valid_actions.append((action, bet_range))
        return valid_actions

    def _select_node(self, node):
        """Select best child node using UCB1 with cached values"""
        best_score = float('-inf')
        best_action = None
        best_child = None

        total_visits = sum(child.visits for child in node.children.values())
        exploration_factor = self.c_param * math.sqrt(math.log(total_visits + 1))

        for action, child in node.children.items():
            if child.visits == 0:
                return action, child

            exploit = child.value / child.visits
            explore = exploration_factor / math.sqrt(child.visits)
            ucb = exploit + explore

            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_node(self, node, observation):
        """Create a new child node with smart bet sizing"""
        if node.untried_actions is None:
            node.untried_actions = self._get_valid_actions(observation)
        
        if not node.untried_actions:
            return None, None

        action_info = node.untried_actions.pop(0)
        action_type, bet_range = action_info
        
        bet_size = 0
        if action_type == PlayerAction.BET and bet_range[1] > 0:
            pot_size = observation[indices.POT_SIZE]
            # Smart bet sizing based on pot
            if pot_size > 0:
                bet_size = min(pot_size * 0.75, bet_range[1])
            else:
                bet_size = bet_range[0]
        
        action = Action(action_type, bet_size)
        new_state = self._get_state_key(observation)
        new_node = MCTSNode(state=new_state, parent=node)
        node.children[action] = new_node
        
        return action, new_node

    def _simulate(self, observation):
        """Quick simulation with state caching"""
        state_key = self._get_state_key(observation)
        if state_key in self.cache:
            return self.cache[state_key]
        
        sim_reward = 0
        valid_actions = self._get_valid_actions(observation)
        
        if not valid_actions:
            return 0
        
        # Enhanced reward calculation
        action_type, bet_range = valid_actions[np.random.randint(len(valid_actions))]
        pot_size = observation[indices.POT_SIZE]
        stack_size = observation[indices.ACTING_PLAYER_STACK_SIZE]
        
        if action_type == PlayerAction.FOLD:
            sim_reward = -min(pot_size * 0.5, stack_size)
        elif action_type == PlayerAction.CHECK:
            sim_reward = 0.1 * pot_size if pot_size > 0 else 0.1
        elif action_type == PlayerAction.CALL:
            call_amount = min(bet_range[0], stack_size)
            pot_odds = call_amount / (pot_size + call_amount + 1e-6)
            sim_reward = 0.2 * (1 - pot_odds) * pot_size
        elif action_type == PlayerAction.BET:
            bet_amount = min(bet_range[1], stack_size, pot_size * 0.75)
            pot_odds = bet_amount / (pot_size + bet_amount + 1e-6)
            sim_reward = 0.3 * pot_odds * pot_size
            
        self.cache[state_key] = sim_reward
        return sim_reward

    def _backpropagate(self, node, reward):
        """Iterative backpropagation"""
        current = node
        depth = 0
        while current is not None and depth < self.max_depth:
            current.visits += 1
            current.value += reward
            current = current.parent
            depth += 1

    def get_action(self, observation):
        """Select best action using MCTS with early stopping"""
        self.observations.append(observation)
        state = self._get_state_key(observation)
        
        if state not in self.nodes:
            self.nodes[state] = MCTSNode(state=state)
        root = self.nodes[state]
        
        # Run simulations with early stopping
        unchanged_count = 0
        best_value = float('-inf')
        for _ in range(self.n_simulations):
            node = root
            sim_obs = observation.copy()
            
            # Selection with depth limit
            depth = 0
            while depth < self.max_depth and node.untried_actions is None or not node.untried_actions:
                if not node.children:
                    break
                action, node = self._select_node(node)
                if node is None:
                    break
                depth += 1
            
            # Expansion and Simulation
            if node is not None and depth < self.max_depth:
                action, new_node = self._expand_node(node, sim_obs)
                if new_node is not None:
                    reward = self._simulate(sim_obs)
                    self._backpropagate(new_node, reward)
                    
                    # Check for convergence
                    max_value = max((child.value/child.visits if child.visits > 0 else float('-inf') 
                                   for child in root.children.values()), default=float('-inf'))
                    if abs(max_value - best_value) < 1e-3:
                        unchanged_count += 1
                        if unchanged_count > 5:  # Early stopping
                            break
                    else:
                        best_value = max_value
                        unchanged_count = 0
        
        # Choose best action
        best_action = None
        best_value = float('-inf')
        
        for action, child in root.children.items():
            if child.visits > 0:
                value = child.value / child.visits
                if value > best_value:
                    best_value = value
                    best_action = action
        
        # Fallback to smart default action
        if best_action is None:
            valid_actions = self._get_valid_actions(observation)
            # Prefer checking to folding when possible
            check_actions = [a for a in valid_actions if a[0] == PlayerAction.CHECK]
            if check_actions:
                action_type, _ = check_actions[0]
                best_action = Action(action_type, 0)
            else:
                action_type, bet_range = valid_actions[0]
                best_action = Action(action_type, bet_range[0] if action_type == PlayerAction.BET else 0)
        
        self.actions.append(best_action)
        return best_action