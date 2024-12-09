import numpy as np
from HANDEVALUATOR import HandEvaluator

def featurize_opponents(opponent_bets):
    """
    Extract betting metrics for all opponents.
    Args:
        opponent_bets (list of lists): Each sublist contains bets made by an opponent.
    Returns:
        list: Aggregated betting metrics for all opponents.
    """
    features = []
    
    for bets in opponent_bets:
        if len(bets) > 3:
            bets_array = np.array(bets)
            mean_bet = np.mean(bets_array)
            
            # Basic statistics
            features.append(np.min(bets_array))
            features.append(np.median(bets_array))
            features.append(mean_bet)
            features.append(np.max(bets_array))
            
            # Betting volatility
            features.append(np.std(bets_array) / (mean_bet + 1e-8))
            
            # Bet progression
            features.append(np.mean(np.diff(bets_array)))
            
            # Recent betting average (last 3 bets)
            features.append(np.mean(bets_array[-3:]))
            
            # Aggression frequency
            aggressive_actions = np.sum(bets_array > mean_bet)
            features.append(aggressive_actions / len(bets_array))
            
        else:
            # Not enough data - pad with zeros
            features.extend([0] * 8)
            
    return np.array(features, dtype=np.float32)

def featurize_state(player_hand, community_cards, opponent_bets):
    """
    Combine hand and opponent metrics into a single state vector.
    Args:
        player_hand (list): Player's private cards.
        community_cards (list): Community cards.
        opponent_bets (list of lists): Each sublist contains bets made by an opponent.
    Returns:
        np.array: Full state vector.
    """
    hand_features = HandEvaluator.featurize_hand(player_hand, community_cards)
    opponent_features = featurize_opponents(opponent_bets)
    return np.concatenate([hand_features, opponent_features])
