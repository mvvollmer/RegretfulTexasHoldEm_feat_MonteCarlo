import numpy as np
from REBELAGENT import REBELAgent
from FEATURIZESTATE import featurize_state
import pokerenv.obs_indices as indices


def learningLoop(table, agents, active_players, n_iterations, batch_size=32, train_interval=1):
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
        print(f"Iteration: {iteration}")
        
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

                # Update observation and acting player
                obs = next_obs
                acting_player = int(obs[indices.ACTING_PLAYER])
            info = []
            for player in table.players:
                if player.stack > 0:
                    info.append((player, player.name))
            if len(info) == 1 or hands == 50:
                roundDone = True
                break
            obs = table.resetHand(info)
            acting_player = int(obs[indices.ACTING_PLAYER])
            done = False
            

        # Track winnings and winners
        starting_stack = table.stack_low
        winnings = [player.stack - starting_stack for player in table.OriginalPlayers]
        playerWinnings.append(winnings)
        maxWinnings = max(playerWinnings[-1])
        winners = []
        for agent, player in zip(agents, table.players):
            if player.stack - starting_stack == maxWinnings:
                playerWins[player_map[player.name]] += 1
                winners.append((agent, player))
            if isinstance(agent, REBELAgent):
                agent.update_exploration(player.stack)  # Update exploration based on performance
                playerCards = player.cards
                nextState = featurize_state(playerCards, communityCards, playerBets)
                if player.stack - starting_stack == maxWinnings:
                    agent.store_transition(state, action, 1, nextState)
                elif player.stack - starting_stack >= 0:
                    agent.store_transition(state, action, player.stack/maxWinnings, nextState)
                else:
                    agent.store_transition(state, action, -1, nextState) 
                
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