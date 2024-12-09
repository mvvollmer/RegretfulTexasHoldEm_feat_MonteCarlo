import numpy as np
import random
from REBELAGENT import REBELAgent
from CREATEENVIORMENT import createEnviorment
from DETERMINISTICAGENTS import RandomAgent, FoldAgent, DeterminedAgent
from FEATURIZESTATE import featurize_state
from pokerenv.obs_indices import indices
import torch


def train_rebel_with_self_play(main_agent, agent_options, n_iterations=2000, batch_size=32, n_players=6):
    """
    Train a REBEL agent through self-play.
    
    Args:
        table: The poker environment/table
        n_iterations: Number of hands to play
        batch_size: Batch size for training
        n_players: Number of players at the table
    
    Returns:
        main_agent: The trained REBEL agent
        training_winnings: List of winnings history
    """
    active_players = 6
    # Create main agent and copies for opponents
    main_agent = REBELAgent(num_players=active_players)
    agents = [main_agent]
    for i in range(n_players - 1):
        agents.append(agent_options[random.choice(range(len(agent_options)))])
    
    # shuffle agents to randomize main agent position
    random.shuffle(agents)
    
    # Get index of main agent
    main_agent_idx = agents.index(main_agent)
    
    player_names = {0: 'Agent1', 1: 'Agent2', 2:'Agent3',3:'Agent4', 4:'Agent5', 5:'Agent6'} # Rest are defaulted to player3, player4...
    player_names[main_agent_idx] = 'MainAgent'
    # Should we only log the 0th players (here TrackedAgent1) private cards to hand history files
    track_single_player = False
    # Bounds for randomizing player stack sizes in reset()
    low_stack_bbs = 499
    high_stack_bbs = 500
    hand_history_location = 'Hands/'
    invalid_action_penalty = 10
    
    table = createEnviorment(active_players, agents, player_names, low_stack_bbs, high_stack_bbs, hand_history_location, invalid_action_penalty, track_single_player)
    
    # Initialize tracking
    training_winnings = [[0 for _ in range(n_players)]]
    playerBets = [[] for _ in range(n_players)]
    
    # Initial state collection for scaler
    initial_states = []
    for _ in range(100):
        table.n_players = n_players
        obs = table.reset()
        acting_player = int(obs[indices.ACTING_PLAYER])
        playerCards = table.players[acting_player].cards
        communityCards = table.cards
        if isinstance(agents[acting_player], REBELAgent):
            state = featurize_state(playerCards, communityCards, playerBets)
            initial_states.append(state)
    
    # Fit scaler for all agents
    for agent in agents:
        if isinstance(agent, REBELAgent):
            agent.fit_scaler(np.array(initial_states))
    
    # Main training loop
    for iteration in range(1, n_iterations + 1):
        # Reset environment
        table.n_players = n_players
        obs = table.reset()
        acting_player = int(obs[indices.ACTING_PLAYER])
        done = False
        
        # Reset agents for new hand
        for agent in agents:
            agent.reset()
        
        # Play one hand
        while not done:
            current_player = table.players[acting_player]
            agent = agents[acting_player]
            playerCards = current_player.cards
            communityCards = table.cards
            if isinstance(agent, REBELAgent):
                state = featurize_state(playerCards, communityCards, playerBets)
            
                # Get action
                action = agent.get_action(state)
            else:
                action = agent.get_action(obs)
                
            next_obs, reward, done, _ = table.step(action)
            
            # Track betting
            playerBet = current_player.bet_this_street
            playerBets[acting_player].append(playerBet)
            
            if not done:
                if isinstance(agent, REBELAgent):
                    # Store transition for the acting agent
                    nextState = featurize_state(playerCards, communityCards, playerBets)
                    agent.store_transition(state, action, 0, nextState)
                
                # Update acting player
                obs = next_obs
                acting_player = int(obs[indices.ACTING_PLAYER])
        
        # Handle end of hand
        hand_winnings = [player.winnings for player in table.players]
        training_winnings.append(hand_winnings)
        max_winnings = max(hand_winnings)
        
        # Update agents based on final hand results
        for agent_idx, (agent, player) in enumerate(zip(agents, table.players)):
            if isinstance(agent, REBELAgent):
                # Calculate final state
                playerCards = player.cards
                state = featurize_state(playerCards, communityCards, playerBets)
                nextState = featurize_state(playerCards, communityCards, playerBets)
                
                # Store final transition with win/loss reward
                if player.winnings == max_winnings:
                    agent.store_transition(state, action, 1, nextState)
                else:
                    agent.store_transition(state, action, -1, nextState)
                
                # Update exploration
                agent.update_exploration(player.winnings)
            
            
        # Train all agents
        if iteration % 1 == 0:  # Can adjust training frequency
            for agent in agents:
                if agent is main_agent:
                    agent.train(batch_size)
        
        # Logging
        if iteration % 500 == 0:
            main_agent_avg_winnings = np.mean([w[main_agent_idx] for w in training_winnings[-500:]])
            print(f"Iteration: {iteration}")
            print(f"Main agent average winnings (last 500 hands): {main_agent_avg_winnings:.2f}")
            
        # Optional: Save checkpoint of main agent
        if iteration % 1000 == 0:
            save_agent(main_agent, f'rebel_agent_checkpoint_{iteration}.pt')
    
    return main_agent, training_winnings

def save_agent(agent, filename):
    """Save the agent's networks and training state"""
    torch.save({
        'value_network': agent.value_network.state_dict(),
        'policy_network': agent.policy_network.state_dict(),
        'bet_value_network': agent.bet_value_network.state_dict(),
        'value_optimizer': agent.value_optimizer.state_dict(),
        'policy_optimizer': agent.policy_optimizer.state_dict(),
        'bet_value_optimizer': agent.bet_value_optimizer.state_dict(),
        'scaler': agent.scaler
    }, filename)