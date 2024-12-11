import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math
from pokerenv.common import PlayerAction, Action

# Neural Network Definitions
class PokerPolicyNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, 4)  # 4 actions
        self.bet_head = nn.Linear(64, 1)     # Bet sizing
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        bet_size = torch.sigmoid(self.bet_head(x))
        return action_probs, bet_size

class PokerValueNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)

class MCTSNode:
    def __init__(self, state=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 1.0
        self.untried_actions = None

class NeuralMCTSAgent:
    def __init__(self, input_size=128, c_param=1.414, n_simulations=50, learning_rate=0.001):
        self.c_param = c_param
        self.n_simulations = n_simulations
        self.actions = []
        self.observations = []
        self.rewards = []
        self.nodes = {}
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PokerPolicyNetwork(input_size).to(self.device)
        self.value_net = PokerValueNetwork(input_size).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Memory for training
        self.memory = []
        
    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def _preprocess_state(self, observation):
        """Convert observation to network input format"""
        # Extract relevant features from observation
        position = observation[0]  # Acting player position
        stack = observation[-2] if len(observation) > 2 else 1000  # Stack size
        pot = observation[1]  # Pot size
        
        # Include cards if available (indices 6-9 for hole cards)
        cards_features = []
        if observation[6] != 0:  # If cards exist
            cards_features = [
                observation[6] / 4, observation[7] / 13,     # Card 1 (normalized)
                observation[8] / 4, observation[9] / 13      # Card 2 (normalized)
            ]
        else:
            cards_features = [0, 0, 0, 0]
        
        # Combine features
        state = np.array([
            position / 6,     # Normalize position
            stack / 1000,    # Normalize stack
            pot / 1000,      # Normalize pot
            *cards_features  # Add normalized card features
        ], dtype=np.float32)
        
        # Convert to tensor here
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
      
    def _get_reward(self, observation, winners):
        """Calculate reward from game outcome"""
        winning_agents = [agent for agent, player in winners]
        winning_players = [player for agent, player in winners]
        
        if self in winning_agents:
            winning_idx = winning_agents.index(self)
            stack = winning_players[winning_idx].stack
            return stack / 1000.0  # Normalize reward
        return -1.0  # Lost hand

    def _get_policy_prediction(self, state):
        """Get action probabilities and bet sizing from policy network"""
        with torch.no_grad():
            network_input = self._preprocess_state(state)
            action_probs, bet_size = self.policy_net(network_input)
            return action_probs.cpu().numpy()[0], bet_size.cpu().numpy()[0]

    def _get_value_prediction(self, state):
        """Get state value prediction from value network"""
        with torch.no_grad():
            network_input = self._preprocess_state(state)
            value = self.value_net(network_input)
            return value.cpu().numpy()[0]

    def _select_node(self, node):
        """Select best child node using PUCT formula"""
        best_score = float('-inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            # PUCT formula
            puct = child.value / (child.visits + 1e-8)
            puct += self.c_param * child.prior * np.sqrt(node.visits) / (1 + child.visits)
            
            if puct > best_score:
                best_score = puct
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_node(self, node, observation):
        """Create a new child node"""
        if node.untried_actions is None:
            node.untried_actions = self._get_valid_actions(observation)
        
        if not node.untried_actions:
            return None, None

        # Get policy prediction
        action_probs, bet_size = self._get_policy_prediction(observation)
        
        # Choose action and create new node
        action_info = node.untried_actions.pop(0)
        action_type, bet_range = action_info
        
        # Create action
        bet_amount = 0
        if action_type == PlayerAction.BET and bet_range[1] > 0:
            # Extract scalar value from bet_size array
            bet_size_scalar = bet_size.item() if isinstance(bet_size, np.ndarray) else bet_size
            bet_amount = float(bet_size_scalar * (bet_range[1] - bet_range[0]) + bet_range[0])
        action = Action(action_type, bet_amount)
        
        # Create new node with policy prior
        new_state = observation
        new_node = MCTSNode(state=new_state, parent=node)
        new_node.prior = action_probs[action_type.value]
        node.children[action] = new_node
        
        return action, new_node

    def _simulate(self, observation):
        """Use value network for simulation instead of rollout"""
        return self._get_value_prediction(observation)

    def _backpropagate(self, node, reward):
        """Update node statistics"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_action(self, observation):
        """Select best action using MCTS"""
        self.observations.append(observation)
        root = MCTSNode(state=observation)
        
        # Run simulations
        for _ in range(self.n_simulations):
            node = root
            current_observation = observation.copy()
            
            # Selection
            while node.untried_actions is None or not node.untried_actions:
                if not node.children:
                    break
                action, node = self._select_node(node)
                if node is None:
                    break
            
            # Expansion
            action, node = self._expand_node(node, current_observation)
            
            # Simulation
            if node is not None:
                value = self._simulate(node.state)
                self._backpropagate(node, value)
        
        # Choose best action
        if not root.children:
            # Fallback to policy network if no simulations were possible
            action_probs, bet_size = self._get_policy_prediction(observation)
            action_type = PlayerAction(np.argmax(action_probs))
            valid_actions = self._get_valid_actions(observation)
            
            for va, bet_range in valid_actions:
                if va == action_type:
                    bet_amount = 0
                    if action_type == PlayerAction.BET and bet_range[1] > 0:
                        # Extract scalar value from bet_size array
                        bet_size_scalar = bet_size.item() if isinstance(bet_size, np.ndarray) else bet_size
                        bet_amount = float(bet_size_scalar * (bet_range[1] - bet_range[0]) + bet_range[0])
                    return Action(action_type, bet_amount)
            
            # If chosen action is not valid, use first valid action
            action_type, bet_range = valid_actions[0]
            return Action(action_type, 0)
        
        # Choose action with most visits
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        self.actions.append(best_action)
        return best_action

    def _get_valid_actions(self, observation):
        """Get list of valid actions and their bet ranges"""
        valid_actions = []
        
        # Check each action individually
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

    def store_transition(self, state, action, reward, next_state):
        """Store transition for training"""
        self.memory.append((state, action, reward, next_state))

    def train(self, batch_size):
        """Train neural networks using stored experiences"""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors
        states_preprocessed = [self._preprocess_state(s).squeeze(0) for s in states]
        next_states_preprocessed = [self._preprocess_state(s).squeeze(0) for s in next_states]
        
        states_tensor = torch.stack(states_preprocessed).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states_preprocessed).to(self.device)
        
        # Get action indices
        action_indices = torch.LongTensor([a.action_type.value for a in actions]).to(self.device)

        # Train policy network
        self.policy_optimizer.zero_grad()
        action_probs, _ = self.policy_net(states_tensor)
        selected_probs = action_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        policy_loss = -torch.mean(torch.log(selected_probs + 1e-10) * rewards_tensor)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()

        # Train value network
        self.value_optimizer.zero_grad()
        values = self.value_net(states_tensor)
        value_loss = F.mse_loss(values.squeeze(), rewards_tensor)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()

    def update_policy(self, winners):
        """Update policy based on game outcome and gathered transitions."""
        if not self.observations or not self.actions:
            return
            
        try:
            # Calculate final reward from game outcome
            final_observation = self.observations[-1]
            reward = self._get_reward(final_observation, winners)
            
            # Store transitions for all actions taken in this hand
            for i, (obs, action) in enumerate(zip(self.observations, self.actions)):
                # Get next observation (if available)
                next_obs = self.observations[i + 1] if i < len(self.observations) - 1 else final_observation
                
                # Store transition with intermediate reward of 0 for non-terminal states
                # and final reward for the terminal state
                transition_reward = reward if i == len(self.observations) - 1 else 0
                self.store_transition(obs, action, transition_reward, next_obs)
            
            # Train networks if we have enough samples
            if len(self.memory) >= 32:
                self.train(batch_size=32)
                
            # Clear the MCTS tree for the next hand
            self.nodes = {}
                
        except Exception as e:
            print(f"Error in MCTS update_policy: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Always clear the episode data
            self.actions = []
            self.observations = []
            self.rewards = []