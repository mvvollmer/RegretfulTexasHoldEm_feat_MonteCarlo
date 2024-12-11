from REBELAGENT import REBELAgent
from FEATURIZESTATE import featurize_state
import pokerenv.obs_indices as indices
from CFRAGENT import CFRAgent
from pokerenv.common import PlayerAction, Action, action_list
import torch
import numpy as np



class HybridPokerAgent:
    def __init__(self, num_players, hidden_size=64, lr=1e-3, gamma=0.999):
        self.rebel = REBELAgent(num_players)
        self.cfr = CFRAgent()
        self.use_cfr_probability = 0.7  # Probability of using CFR vs REBEL
        self.cfr_weight = 0.5  # Weight given to CFR action values
        self.strategy_update_frequency = 1  # How often to update CFR strategy
        self.training_step = 100
        
    def get_action(self, observation, state):
        """Hybrid decision making combining REBEL and CFR approaches"""
        # First get CFR action using raw observation
        cfr_action = self.cfr.get_action(observation)
        
        # Get CFR strategy for the current info set
        info_set = self.cfr._get_info_set(observation)
        cfr_strategy = self.cfr.get_strategy(info_set)
        
        # Use the featurized state for REBEL
        with torch.no_grad():
            # Make sure state has the correct dimensions
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.rebel.device)
            rebel_action = self.rebel.get_action(state)
            rebel_probs = self.rebel.policy_network(state_tensor).softmax(dim=-1).cpu().numpy()[0]
        
        # Combine strategies
        combined_strategy = (1 - self.cfr_weight) * rebel_probs + self.cfr_weight * cfr_strategy
        
        # Choose final action based on combined strategy
        if np.random.random() < self.use_cfr_probability:
            final_action = cfr_action
        else:
            final_action = rebel_action
            
        # If betting, use REBEL's bet sizing with CFR influence
        if final_action.action_type == PlayerAction.BET:
            rebel_bet = self.rebel.bet_value_network(state_tensor).item()
            cfr_bet = cfr_action.bet_amount
            final_bet = (1 - self.cfr_weight) * rebel_bet + self.cfr_weight * cfr_bet
            final_action = Action(PlayerAction.BET, final_bet)
            
        return final_action
    
    def _update_cfr_strategy(self, featurized_state):
        """Update CFR strategy using accumulated experience"""
        # Skip if no observations
        if not self.cfr.observations:
            return
            
        for obs, action, reward in zip(self.cfr.observations, self.cfr.actions, self.cfr.rewards):
            info_set = self.cfr._get_info_set(obs)
            
            # Calculate counterfactual values for other actions
            other_action_values = np.zeros(4)
            
            # Get community cards
            community = []
            for i in range(5):
                if obs[16 + i*2] != 0:
                    suit = int(obs[16 + i*2])
                    rank = int(obs[17 + i*2])
                    community.append((suit, rank))
                    
            player_bets = []  # Extract player bets based on your observation structure
            # You'll need to adjust these indices based on your observation structure
            for i in range(self.rebel.num_players):
                bet_index = 26 + i  # Adjust this index based on your observation structure
                player_bets.append(obs[bet_index] if bet_index < len(obs) else 0)
            
            for a in range(4):
                if a == action.action_type.value:
                    other_action_values[a] = reward
                else:
                    # Use featurized state for REBEL's value network
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(featurized_state).unsqueeze(0).to(self.rebel.device)
                        other_action_values[a] = self.rebel.value_network(state_tensor).item()
            
            # Update CFR regrets and strategy
            self.cfr.update_regret(info_set, action, reward, other_action_values)
            strategy = self.cfr.get_strategy(info_set)
            self.cfr.update_strategy(info_set, strategy)

    def store_transition(self, state, action, reward, next_state):
        """Store experience in both agents"""
        # Store in REBEL agent
        self.rebel.store_transition(state, action, reward, next_state)
        
        # Store reward for CFR (we already stored observation and action in get_action)
        if hasattr(self.cfr, 'rewards'):
            self.cfr.rewards.append(reward)
        
        self.training_step += 1
        if self.training_step % self.strategy_update_frequency == 0:
            self._update_cfr_strategy(state)
        
    def train(self, batch_size=32):
        """Train the REBEL component"""
        self.rebel.train(batch_size)
        
        # Adjust CFR influence based on performance
        if len(self.rebel.recent_rewards) >= 20:
            avg_reward = np.mean(list(self.rebel.recent_rewards)[-20:])
            if avg_reward > 0.2:
                self.cfr_weight = max(0.2, self.cfr_weight * 0.95)  # Reduce CFR influence
            elif avg_reward < -0.1:
                self.cfr_weight = min(0.6, self.cfr_weight * 1.05)  # Increase CFR influence
        
    def reset(self):
        """Reset both agents"""
        self.rebel.reset()
        self.cfr.reset()