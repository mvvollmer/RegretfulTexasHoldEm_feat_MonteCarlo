from REBELAGENT import REBELAgent
from FEATURIZESTATE import featurize_state
import pokerenv.obs_indices as indices
from CFRAGENT import CFRAgent
from pokerenv.common import PlayerAction, Action, action_list
import torch
import numpy as np
from NETWORKS import ValueNetwork, PolicyNetwork, BetValueNetwork
from torch.optim import Adam
from collections import defaultdict
import torch.nn.functional as F
from REPLAYBUFFER import ReplayBuffer



class ImprovedREBELAgent:
    def __init__(self, num_players, hidden_size=64, lr=1e-3, gamma=0.999, cfr_iterations=10):
        # Network initialization
        self.state_dim = 21 + 8 * num_players  # Using same state dim as original REBEL
        self.value_network = ValueNetwork(self.state_dim, hidden_size, 1)
        self.policy_network = PolicyNetwork(self.state_dim, hidden_size, 4)
        self.bet_value_network = BetValueNetwork(self.state_dim, hidden_size)
        
        # CFR parameters
        self.cfr_iterations = cfr_iterations
        self.max_depth = 3  # Depth limit for subgame solving
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimization setup
        self.value_optimizer = Adam(self.value_network.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.policy_network.parameters(), lr=lr)
        self.bet_optimizer = Adam(self.bet_value_network.parameters(), lr=lr)
        
        # Storage for CFR
        self.regret_sum = defaultdict(lambda: np.zeros(4))  # 4 actions
        self.strategy_sum = defaultdict(lambda: np.zeros(4))
        self.leaf_nodes = set()
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=50000)
        
        self.num_players = num_players
        
        # Exploration parameters
        self.epsilon = 0.4
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
    def solve_subgame(self, root_state, max_depth=None):
        """
        Solve a depth-limited subgame using CFR-D starting from root_state
        """
        if max_depth is None:
            max_depth = self.max_depth
            
        # Convert root_state to tuple for dictionary keys
        root_tuple = tuple(root_state)
            
        policy_sum = defaultdict(lambda: np.zeros(4))
        value_sum = defaultdict(float)
        
        # Clear leaf nodes for this subgame
        self.leaf_nodes = set()
        
        for t in range(self.cfr_iterations):
            # Initialize reach probabilities
            reach_probs = np.ones(4)
            
            # Forward pass with reach probabilities
            curr_policy = self._cfr_forward(root_state, reach_probs, max_depth)
            
            # Get value estimates for leaf nodes
            leaf_values = self._update_leaf_values()
            
            # Backward pass to compute values and update regrets
            values = self._cfr_backward(root_state, curr_policy, leaf_values, reach_probs)
            
            # Accumulate average policy and values
            for info_state, policy in curr_policy.items():
                policy_sum[info_state] += policy * reach_probs
            value_sum[root_tuple] += values[tuple(root_state)]  # Use tuple for dictionary key
            
        # Average over iterations
        avg_policy = {s: policy_sum[s]/self.cfr_iterations for s in policy_sum}
        avg_value = value_sum[root_tuple]/self.cfr_iterations
        
        return avg_policy, avg_value
    
    def _cfr_forward(self, state, reach_probs, depth):
        """Forward pass of CFR-D"""
        # Convert state to hashable type (tuple)
        state_tuple = tuple(state)
        
        if depth == 0 or self._is_terminal(state):
            self.leaf_nodes.add(state_tuple)
            return {}
            
        curr_policy = {}
        
        # Get current strategy for this information state
        strategy = self._get_strategy(state_tuple)
        curr_policy[state_tuple] = strategy
        
        # Recurse on all actions with updated reach probabilities
        for action in range(4):
            if strategy[action] > 0:
                next_state = self._get_next_state(state, action)
                next_reach = reach_probs * strategy[action]
                child_policy = self._cfr_forward(next_state, next_reach, depth - 1)
                curr_policy.update(child_policy)
                
        return curr_policy
        
    def _cfr_backward(self, state, policy, leaf_values, reach_probs):
        """Backward pass of CFR-D to compute counterfactual values and update regrets"""
        state_tuple = tuple(state)
        
        if state_tuple in self.leaf_nodes:
            return {state_tuple: leaf_values[state_tuple]}
            
        values = defaultdict(float)
        action_values = np.zeros(4)
        
        # Convert reach_probs to scalar if it's an array
        reach_prob_value = float(np.mean(reach_probs)) if isinstance(reach_probs, np.ndarray) else float(reach_probs)
        
        # Compute counterfactual values for each action
        for action in range(4):
            if policy[state_tuple][action] > 0:
                next_state = self._get_next_state(state, action)
                next_state_tuple = tuple(next_state)
                # Update reach probability calculation
                next_reach = reach_prob_value * policy[state_tuple][action]
                child_values = self._cfr_backward(next_state, policy, leaf_values, next_reach)
                action_values[action] = child_values[next_state_tuple]
                values[state_tuple] += policy[state_tuple][action] * action_values[action]
                    
        # Update regrets using scalar reach probability
        for action in range(4):
            regret = action_values[action] - values[state_tuple]
            # Use scalar multiplication for regret update
            self.regret_sum[state_tuple][action] += reach_prob_value * regret
            
        return values
    
    def _get_strategy(self, state):
        """Get current strategy for state using regret matching"""
        regrets = np.maximum(self.regret_sum[state], 0)
        sum_regrets = np.sum(regrets)
        
        if sum_regrets > 0:
            return regrets / sum_regrets
        return np.ones(4) / 4
    
    def _update_leaf_values(self):
        """Estimate values for leaf nodes using value network"""
        leaf_values = {}
        with torch.no_grad():
            for state in self.leaf_nodes:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                value = self.value_network(state_tensor).item()
                leaf_values[state] = value
        return leaf_values
    
    def get_action(self, observation, state):
        """Choose action using subgame solving and networks"""
        # Convert observation to our featurized state
        featurized_state = state  # Since we're using featurized state as PBS
        
        # Epsilon exploration
        if np.random.random() < self.epsilon:
            # Get valid actions from observation
            valid_actions = []
            for i in range(4):
                if observation[1 + i] == 1:  # Valid actions start at index 1
                    valid_actions.append(i)
            
            # Random choice among valid actions
            action_type = PlayerAction(np.random.choice(valid_actions))
            if action_type == PlayerAction.BET:
                bet_size = np.random.uniform(
                    observation[indices.VALID_BET_LOW],
                    observation[indices.VALID_BET_HIGH]
                )
                return Action(action_type, bet_size)
            return Action(action_type, 0)
        
        # Regular action selection
        # Solve depth-limited subgame
        policy, _ = self.solve_subgame(featurized_state)
        
        # Get state tuple for dictionary lookup
        state_tuple = tuple(featurized_state)
        
        # Get action probabilities
        if state_tuple not in policy:
            # Create logits for valid actions
            logits = np.zeros(4) - np.inf  # Initialize to -inf for invalid actions
            for i in range(4):
                if observation[1 + i] == 1:
                    logits[i] = 0  # Set to 0 for valid actions
            action_probs = F.softmax(torch.FloatTensor(logits), dim=0).numpy()
        else:
            # Use policy logits and mask invalid actions
            logits = np.log(policy[state_tuple] + 1e-10)  # Add small constant for numerical stability
            for i in range(4):
                if observation[1 + i] != 1:
                    logits[i] = -np.inf
            action_probs = F.softmax(torch.FloatTensor(logits), dim=0).numpy()
        
        # Choose action type (no need for additional normalization)
        action_type = PlayerAction(np.random.choice(4, p=action_probs))
        
        # If betting, use bet value network with valid bounds
        if action_type == PlayerAction.BET:
            state_tensor = torch.FloatTensor(featurized_state).unsqueeze(0).to(self.device)
            bet_value = self.bet_value_network(state_tensor).item()
            
            # Clip bet to valid range
            bet_value = np.clip(
                bet_value,
                observation[indices.VALID_BET_LOW],
                observation[indices.VALID_BET_HIGH]
            )
            return Action(action_type, bet_value)
        
        return Action(action_type, 0)
        
    def train_batch(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
                
        # Sample batch and unpack components
        batch_data = self.replay_buffer.sample(batch_size)
        states, actions, rewards = batch_data[0], batch_data[1], batch_data[2]
        
        # Process states (which is already a proper array)
        state_tensor = torch.FloatTensor(states).to(self.device)
        
        # Convert actions and rewards to tensors
        action_tensor = torch.LongTensor(actions).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Get CFR solutions for states 
        cfr_policies = []
        cfr_values = []
        for state in states:
            state_tuple = tuple(state)
            policy, value = self.solve_subgame(state, max_depth=2)
            cfr_policies.append(policy.get(state_tuple, np.ones(4)/4))
            cfr_values.append(value)
                
        cfr_policy_tensor = torch.FloatTensor(np.array(cfr_policies)).to(self.device)
        cfr_value_tensor = torch.FloatTensor(cfr_values).to(self.device)
        
        # Train value network
        value_pred = self.value_network(state_tensor)
        value_loss = F.mse_loss(value_pred, reward_tensor.unsqueeze(1))  # Add TD learning
        cfr_value_loss = F.mse_loss(value_pred, cfr_value_tensor.unsqueeze(1))
        total_value_loss = 0.5 * (value_loss + cfr_value_loss)
        
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Train policy network combining TD and CFR targets
        policy_pred = self.policy_network(state_tensor)
        action_onehot = F.one_hot(action_tensor, num_classes=4).float()
        policy_targets = 0.7 * action_onehot + 0.3 * cfr_policy_tensor
        
        policy_loss = F.kl_div(
            F.log_softmax(policy_pred, dim=1),
            policy_targets,
            reduction='batchmean'
        )
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
    
    def train(self, batch_size):
        """Wrapper for train_batch to match the expected interface"""
        self.train_batch(batch_size)
        
    def _get_next_state(self, state, action):
        """
        Predict next state given current state and action.
        Uses value network to estimate state value changes.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            current_value = self.value_network(state_tensor)
            
            # Estimate state changes based on action
            if action == PlayerAction.FOLD.value:
                value_change = -1.0
            elif action == PlayerAction.CHECK.value:
                value_change = 0.0
            elif action == PlayerAction.CALL.value:
                value_change = -0.2
            else:  # BET
                bet_value = self.bet_value_network(state_tensor).item()
                value_change = -bet_value / 100.0  # Normalize bet impact
                
            # Update state features based on action
            next_state = state.copy()
            next_state[action] = 1  # Mark action taken
            next_state[-1] = float(current_value + value_change)  # Update value estimate
            
            return next_state
            
    def _is_terminal(self, state):
        """
        Check if state is terminal.
        In poker, terminal states are when:
        1. Only one player remains (others folded)
        2. River card dealt and all betting complete
        3. All players are all-in
        """
        # Extract relevant features from state
        street_index = -5  # Adjust based on your state representation
        num_active_players = sum(state[8:8+self.num_players] > 0)
        is_river = state[street_index] == 1
        all_actions_complete = all(state[1:5] == 0)  # No more valid actions
        
        return (num_active_players == 1 or 
                (is_river and all_actions_complete) or
                all(state[8:8+self.num_players] == state[8]))  # All equal bets
                
    def update_exploration(self, reward):
        """Update exploration rate based on performance"""
        if reward > 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = min(0.4, self.epsilon * 1.05)
            
    def store_transition(self, state, action, reward, next_state):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done=False)
        
    def reset(self):
        """Reset episode-specific variables"""
        # Clear temporary CFR storage
        self.leaf_nodes.clear()
        self.strategy_sum.clear()
        
        # Optional: clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
