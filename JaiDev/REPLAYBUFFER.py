import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.
        Args:
            capacity (int): Maximum number of transitions to store
            alpha (float): Priority exponent for importance sampling
            beta (float): Initial importance sampling weight
            beta_increment (float): Increment for beta after each sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to prevent zero probabilities
        
    def add(self, state, action, reward, next_state, done):
        """Add a transition with maximum priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions with prioritization.
        Returns:
            tuple: Batches of (states, actions, rewards, next_states, dones, importance_weights, indices)
        """
        total = len(self.buffer)
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(total, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Increment beta for next time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array([a.action_type.value for a in actions]),
            np.array([a.bet_amount for a in actions]),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            weights,
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)