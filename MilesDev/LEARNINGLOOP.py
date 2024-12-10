import numpy as np
from REBELAGENT import REBELAgent
from FEATURIZESTATE import featurize_state
import pokerenv.obs_indices as indices
from HybridPokerAgent import HybridPokerAgent
from IMPROVEDREBELAGENT import ImprovedREBELAgent


def learningLoop(table, agents, active_players, n_iterations, batch_size=32, train_interval=3, max_hands = 30):
    """
    Main loop for training agents in the poker environment.
    
    Args:
        table: The poker environment/table.
        agents: List of agents, including REBELAgent.
        active_players: Number of active players at the table.
        n_iterations: Number of hands to play.
        batch_size: Batch size for training the agents.
        train_interval: Number of hands between training phases.

    Returns:
        playerWinnings: List of winnings for each player after each hand.
    """
    player_map = {player.name: i for i, player in enumerate(table.players)}
    iteration = 1
    playerWinnings = [[0 for _ in table.players]]
    playerWins = [0 for _ in table.players]
    playerBets = [[] for _ in table.players]
    
    initial_states = []
    for _ in range(100):  # Collect 100 initial observations
        table.n_players = active_players
        obs = table.reset()
        acting_player = int(obs[indices.ACTING_PLAYER])
        for agent in agents:
            if isinstance(agent, REBELAgent):
                playerCards = table.players[acting_player].cards  # Replace with correct player handling
                communityCards = table.cards
                state = featurize_state(playerCards, communityCards, playerBets)
                initial_states.append(state)
                
            

    # Fit scalers for all agents
    for agent in agents:
        if isinstance(agent, REBELAgent):
            agent.fit_scaler(np.array(initial_states))

    while iteration <= n_iterations:
        # Reset table and agents
        obs = table.reset()
        table.n_players = active_players
        done = False
        
        for agent in agents:
            agent.reset()
            
            
        roundDone = False
        hands = 0
        while not roundDone:
            while not done:
                current_player = table.players[table.next_player_i]
                agent = agents[player_map[current_player.name]]
                playerCards = current_player.cards
                communityCards = table.cards
                    

                # Fetch action from the agent
                if isinstance(agent, REBELAgent):
                    state = featurize_state(playerCards, communityCards, playerBets)
                    action = agent.get_action(state)
                elif isinstance(agent, HybridPokerAgent):
                    Hstate = featurize_state(playerCards, communityCards, playerBets)
                    action = agent.get_action(obs, Hstate)
                elif isinstance(agent, ImprovedREBELAgent):  # Add this new condition
                    Istate = featurize_state(playerCards, communityCards, playerBets)
                    action = agent.get_action(obs, Istate)
                else:
                    action = agent.get_action(obs)  
                
                # Step the environment
                next_obs, reward, done, _ = table.step(action)

                # Track betting data
                playerBet = current_player.bet_this_street
                playerBets[player_map[current_player.name]].append(playerBet)
                
                if done:
                    hands += 1
                    break
                
                if isinstance(agent, REBELAgent):
                    # Calculate the next state for the REBEL agent
                    nextState = featurize_state(playerCards[0:2], communityCards, playerBets)

                    # Store transition in the agent's replay buffer
                    agent.store_transition(state, action, 0, nextState)
                elif isinstance(agent, HybridPokerAgent):
                    Hnext_state = featurize_state(playerCards, communityCards, playerBets)
                    agent.store_transition(Hstate, action, 0, Hnext_state)
                    
                elif isinstance(agent, ImprovedREBELAgent):  # Add this new condition
                    Inext_state = featurize_state(playerCards, communityCards, playerBets)
                    agent.store_transition(Istate, action, 0, Inext_state)


                # Update observation and acting player
                obs = next_obs
                acting_player = int(obs[indices.ACTING_PLAYER])
            info = []
            for player in table.players:
                if player.stack > 0:
                    info.append((player, player.name))
            if len(info) == 1 or hands == max_hands:
                roundDone = True
                break
            obs = table.resetHand(info)
            acting_player = int(obs[indices.ACTING_PLAYER])
            done = False
            

        # Track winnings and winners
        starting_stack = table.stack_low
        winnings = [player.stack - starting_stack for player in table.OriginalPlayers if player.has_acted ]
        playerWinnings.append(winnings)
        maxWinnings = max(playerWinnings[-1])
        winners = []
        for agent, player in zip(agents, table.players):
            if player.stack - starting_stack == maxWinnings:
                playerWins[player_map[player.name]] += 1
                winners.append((agent, player))
            if isinstance(agent, REBELAgent):
                playerCards = player.cards
                nextState = featurize_state(playerCards, communityCards, playerBets)
                if player.stack - starting_stack == maxWinnings:
                    agent.store_transition(state, action, 1, nextState)
                    agent.update_exploration(1)
                    agent.wins += 1
                elif player.stack - starting_stack >= 0:
                    agent.store_transition(state, action, (player.stack - starting_stack)/maxWinnings, nextState)
                    agent.update_exploration((player.stack - starting_stack)/maxWinnings)
                else:
                    agent.store_transition(state, action, -1, nextState) 
                    agent.update_exploration(-1)
                
                
            elif isinstance(agent, HybridPokerAgent):
                playerCards = player.cards
                Hnext_state = featurize_state(playerCards, communityCards, playerBets)
                if player.stack - starting_stack == maxWinnings:
                    agent.store_transition(Hstate, action, 1, Hnext_state)
                elif player.stack - starting_stack >= 0:
                    agent.store_transition(Hstate, action, (player.stack - starting_stack)/maxWinnings, Hnext_state)
                else:
                    agent.store_transition(Hstate, action, -1, Hnext_state)
                    
            elif isinstance(agent, ImprovedREBELAgent):  # Add this new condition
                playerCards = player.cards
                Inext_state = featurize_state(playerCards, communityCards, playerBets)
                if player.stack - starting_stack == maxWinnings:
                    agent.store_transition(Istate, action, 1, Inext_state)
                    agent.update_exploration(1)
                elif player.stack - starting_stack >= 0:
                    agent.store_transition(Istate, action, (player.stack - starting_stack)/maxWinnings, Inext_state)
                    agent.update_exploration((player.stack - starting_stack)/maxWinnings)
                else:
                    agent.store_transition(Istate, action, -1, Inext_state)
                    agent.update_exploration(-1)
                        
            # Train agents
            if iteration % train_interval == 0:
                for agent in agents:
                    if hasattr(agent, "train"):
                        agent.train(batch_size)
                    
            if hasattr(agent, "update_policy"):
                agent.update_policy(winners)

        # Disable hand history logging
        table.hand_history_enabled = False

        # Print progress every 250 iterations
        if iteration % 50 == 0:
            print(f"Iteration: {iteration}")

        iteration += 1

    return playerWinnings