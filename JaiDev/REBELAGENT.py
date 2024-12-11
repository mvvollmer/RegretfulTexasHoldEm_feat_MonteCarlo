import numpy as np
import random
from pokerenv.common import PlayerAction, Action, action_list

from pokerenv.utils import pretty_print_hand, approx_gt, approx_lte
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import huber_loss, cross_entropy
from collections import deque

from NETWORKS import ValueNetwork, PolicyNetwork, BetValueNetwork
from REPLAYBUFFER import ReplayBuffer


class REBELAgent:
    def __init__(self, num_players, hidden_size=64, lr=1e-3, gamma=0.995):
        # Define game-specific constants
        self.ACTIONS = [PlayerAction.CHECK, PlayerAction.FOLD, PlayerAction.BET, PlayerAction.CALL]
        self.ACTION_MAP = {
            0: (PlayerAction.CHECK, 0),
            1: (PlayerAction.FOLD, 0),
            2: (PlayerAction.BET, 0),
            3: (PlayerAction.CALL, 0)
        }
        
        state_dim = 21 + 8 * num_players  
        self.state_dim = state_dim
        self.gamma = gamma
        self.num_players = num_players

        # Improved network initialization
        self.value_network = ValueNetwork(state_dim, hidden_size, 1)
        self.policy_network = PolicyNetwork(state_dim, hidden_size, 4)
        self.bet_value_network = BetValueNetwork(state_dim, hidden_size)
        
        # More robust optimization parameters
        self.importance_threshold = 0.001
        self.td_error_clip = 1.0
        self.batch_processing_size = 512
        self.min_train_samples = 100  # Increased for better initial learning
        self.grad_clip_norm = 1.0
        
        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value_network.to(self.device)
        self.policy_network.to(self.device)
        self.bet_value_network.to(self.device)
        
        # Optimizers 
        self._setup_optimizers(lr)

        # Enhanced replay buffer
        self.replay_buffer = ReplayBuffer(capacity=50000)  # Increased capacity for better learning
        self.scaler = StandardScaler()
        
        # Training settings
        self.train_counter = 0
        self.update_frequency = 5
        
        # Initialize scaler
        dummy_data = np.zeros((1, state_dim))
        self.scaler.fit(dummy_data)
        
        # Improved learning rate scheduling
        self.lr_scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer, T_max=2000, eta_min=1e-5
        )
        self.lr_scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=2000, eta_min=1e-5
        )
        
        # Better exploration parameters aligned with ReBeL paper
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Slower decay
        self.exploration_bonus = 0.1
        
        # Enhanced performance tracking
        self.recent_rewards = deque(maxlen=10000)
        self.win_streak = 0
        self.running_reward = 0
        self.best_reward = float('-inf')
        
    def _setup_optimizers(self, lr):
        """Initialize optimizers with improvements from ReBeL paper"""
        self.value_optimizer = Adam(self.value_network.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.policy_network.parameters(), lr=lr)
        self.bet_value_optimizer = Adam(self.bet_value_network.parameters(), lr=lr)


    def fit_scaler(self, states):
        """
        Fit the scaler with a batch of states.
        Args:
            states: A batch of state observations.
        """
        self.scaler.fit(states)

    def get_action(self, observation):
        """
        Decide the action based on the observation.
        Outputs both the action type and the bet value if applicable.
        """
        if random.random() < self.epsilon:
            action_type = random.choice(self.ACTIONS)  # Use class constant
            if action_type == PlayerAction.BET:
                bet_value = random.uniform(0, 100)
                return Action(action_type, bet_value)
            return Action(action_type, 0)
        
        with torch.no_grad():  # Keep this for safety
            # Pre-allocate observation array
            observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
            observation = self.scaler.transform(observation)[0]
            
            # More efficient tensor creation
            state_tensor = torch.from_numpy(observation).to(self.device, non_blocking=True).float().unsqueeze(0)
            
            # Get action probabilities more efficiently
            action_probs = self.policy_network(state_tensor)
            action_probs = torch.softmax(action_probs, dim=-1)  # More stable than manual normalization
            action_probs = action_probs.cpu().numpy().flatten()
            
            # Faster validation check
            if not np.all(np.isfinite(action_probs)):
                return Action(PlayerAction.FOLD, 0)
            
            # Temperature scaling without recreation of array
            temperature = max(0.5, self.epsilon)
            np.power(action_probs, 1/temperature, out=action_probs)
            action_probs /= action_probs.sum()
            
            chosen_action = np.random.choice(4, p=action_probs)
            
            if chosen_action == 2:  # BET
                bet_tensor = self.bet_value_network(state_tensor)
                if torch.isnan(bet_tensor).any() or not torch.isfinite(bet_tensor).all():
                    return Action(PlayerAction.CHECK, 0)
                
                bet_value = float(max(0, bet_tensor.cpu().item()))
                return Action(PlayerAction.BET, bet_value)
            
            # Use class constant
            action_type, value = self.ACTION_MAP[chosen_action]
            return Action(action_type, value)
        
    def store_transition(self, state, action, reward, next_state):
        """Store transition with done flag (assumed False during game)"""
        self.replay_buffer.add(state, action, reward, next_state, done=False)
        
    def update_exploration(self, reward):
        """Enhanced exploration update based on ReBeL paper"""
        self.recent_rewards.append(reward)
        self.running_reward = 0.95 * self.running_reward + 0.05 * reward
        
        if self.running_reward > self.best_reward:
            self.best_reward = self.running_reward
        
        # Update epsilon with better heuristics
        if len(self.recent_rewards) >= 30:
            avg_reward = np.mean(list(self.recent_rewards)[-30:])
            if avg_reward < -0.2:
                self.epsilon = min(0.3, self.epsilon * 1.01)
            else:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Enhanced win streak handling
        if reward > 0:
            self.win_streak += 1
            if self.win_streak > 5:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.95)
        else:
            self.win_streak = 0
            if len(self.recent_rewards) >= 10 and np.mean(list(self.recent_rewards)[-10:]) < -0.5:
                self.epsilon = min(0.3, self.epsilon * 1.05)


    def train(self, batch_size=32):
        self.train_counter += 1
        if self.train_counter % self.update_frequency != 0:
            return
        
        if len(self.replay_buffer) < batch_size:
            return

        # Get batch efficiently
        sample_size = min(self.batch_processing_size, len(self.replay_buffer))
        states, actions, bet_values, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(sample_size)

        # Early exit check
        mask = weights > self.importance_threshold
        if not np.any(mask):
            return

        # Pre-allocate tensors once
        device_tensors = {
            'states': torch.FloatTensor(states[mask]).to(self.device),
            'actions': torch.LongTensor(actions[mask]).to(self.device),
            'bet_values': torch.FloatTensor(bet_values[mask]).to(self.device),
            'rewards': torch.FloatTensor(rewards[mask]).to(self.device),
            'next_states': torch.FloatTensor(next_states[mask]).to(self.device),
            'weights': torch.FloatTensor(weights[mask]).to(self.device)
        }

        # Value network update
        with torch.no_grad():
            next_values = self.value_network(device_tensors['next_states']).squeeze()
            target_values = device_tensors['rewards'] + self.gamma * next_values

        current_values = self.value_network(device_tensors['states']).squeeze()
        td_errors = torch.clamp(target_values - current_values, -self.td_error_clip, self.td_error_clip)
        
        value_loss = (device_tensors['weights'] * td_errors ** 2).mean()
        self.value_optimizer.zero_grad(set_to_none=True)  # More efficient than regular zero_grad
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()

        # Update priorities once
        numpy_td_errors = td_errors.detach().abs().cpu().numpy()
        self.replay_buffer.update_priorities(indices[mask], numpy_td_errors)

        # Policy network update with pre-computed advantages
        advantages = td_errors.detach()
        action_probs = self.policy_network(device_tensors['states'])
        action_log_probs = torch.log(action_probs.gather(1, device_tensors['actions'].unsqueeze(-1))).squeeze()
        policy_loss = -(device_tensors['weights'] * advantages * action_log_probs).mean()
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()

        # Bet value network update
        bet_prediction = self.bet_value_network(device_tensors['states']).squeeze()
        bet_value_loss = (device_tensors['weights'] * (bet_prediction - device_tensors['bet_values']) ** 2).mean()
        
        self.bet_value_optimizer.zero_grad(set_to_none=True)
        bet_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.bet_value_network.parameters(), 1.0)
        self.bet_value_optimizer.step()

        # Compute novelty only if needed
        with torch.no_grad():
            state_novelty = self.compute_state_novelty(device_tensors['states'])
            advantages = advantages + self.exploration_bonus * state_novelty

        # Single update of learning rates
        self.lr_scheduler_value.step()
        self.lr_scheduler_policy.step()
        
    def compute_state_novelty(self, states):
        """Improved state novelty computation with better numerical stability"""
        with torch.no_grad():
            features = self.value_network.get_features(states)
            features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
            distances = torch.cdist(features, features)
            novelty = distances.mean(dim=1)
            novelty = novelty / (novelty.max() + 1e-8)
            return novelty

        
    def reset(self):
        # Reset performance tracking for this episode
        self.win_streak = 0
    
        # Keep a small running reward memory but decay it
        self.running_reward *= 0.5
    
        # Optional: Clear GPU memory if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
