from treys import Evaluator, Card
from treys.lookup import LookupTable
import numpy as np

evaluator = Evaluator()

class HandEvaluator:
    # Class-level constants
    RANK_CLASSES = {
        LookupTable.MAX_HIGH_CARD: 0,
        LookupTable.MAX_PAIR: 1,
        LookupTable.MAX_TWO_PAIR: 2,
        LookupTable.MAX_THREE_OF_A_KIND: 3,
        LookupTable.MAX_STRAIGHT: 4,
        LookupTable.MAX_FLUSH: 5,
        LookupTable.MAX_FULL_HOUSE: 6,
        LookupTable.MAX_FOUR_OF_A_KIND: 7,
        LookupTable.MAX_STRAIGHT_FLUSH: 8,
        LookupTable.MAX_ROYAL_FLUSH: 9
    }

    @staticmethod
    def featurize_hand(player_hand, community_cards):
        """
        Enhanced hand strength metrics with relative strength indicators.
        
        Returns:
            np.array: Features array containing:
                [0-9]: Hand type indicators
                [10-12]: Draw potentials (flush, straight, full house)
                [13]: High card relative strength (0-1)
                [14]: Pair relative strength (0-1)
                [15]: Two pair relative strength (0-1)
                [16]: Trips relative strength (0-1)
                [17]: Straight relative strength (0-1)
                [18]: Flush relative strength (0-1)
                [19]: Full house relative strength (0-1)
                [20]: Four of a kind relative strength (0-1)
        """
        features = np.zeros(21, dtype=np.float32)
        
        # Pre-flop handling
        if not community_cards:
            return HandEvaluator._preflop_features(player_hand)
            
        # Get hand rank and class
        hand_score = evaluator.evaluate(player_hand, community_cards)
        rank_class = evaluator.get_rank_class(hand_score)
        
        # Basic hand type indicators (0-9)
        for max_val, idx in HandEvaluator.RANK_CLASSES.items():
            if rank_class == LookupTable.MAX_TO_RANK_CLASS[max_val]:
                features[idx] = 1
                break
        
        # Get draw potentials if not on river
        if len(community_cards) < 5:
            features[10:13] = HandEvaluator._calculate_draw_potentials(player_hand, community_cards)
        
        # Calculate relative strength metrics
        all_cards = player_hand + community_cards
        ranks = [Card.get_rank_int(card) for card in all_cards]
        suits = [Card.get_suit_int(card) for card in all_cards]
        
        # Get relative strength based on hand type
        if features[0]:  # High card
            features[13] = HandEvaluator._high_card_strength(player_hand)
            
        elif features[1]:  # Pair
            features[14] = HandEvaluator._pair_strength(ranks, player_hand)
            
        elif features[2]:  # Two pair
            features[15] = HandEvaluator._two_pair_strength(ranks, player_hand)
            
        elif features[3]:  # Three of a kind
            features[16] = HandEvaluator._trips_strength(ranks, player_hand)
            
        elif features[4]:  # Straight
            features[17] = HandEvaluator._straight_strength(ranks)
            
        elif features[5]:  # Flush
            features[18] = HandEvaluator._flush_strength(all_cards)
            
        elif features[6]:  # Full house
            features[19] = HandEvaluator._full_house_strength(ranks, player_hand)
            
        elif features[7]:  # Four of a kind
            features[20] = HandEvaluator._four_kind_strength(ranks, player_hand)
            
        return features
    
    @staticmethod
    def _preflop_features(hole_cards):
        """Efficient pre-flop feature calculation"""
        features = np.zeros(21, dtype=np.float32)
        
        # Get ranks and suits
        ranks = sorted([Card.get_rank_int(card) for card in hole_cards])
        suits = [Card.get_suit_int(card) for card in hole_cards]
        
        # Basic hand features
        is_pair = ranks[0] == ranks[1]
        is_suited = suits[0] == suits[1]
        
        # Set appropriate feature
        if is_pair:
            features[1] = 1  # Pair
        else:
            features[0] = 1  # High card
            
        # Draw potentials for pre-flop
        features[10] = float(is_suited)  # Flush draw potential
        features[11] = float(abs(ranks[0] - ranks[1]) <= 4)  # Straight draw potential
        features[12] = 0  # No full house potential pre-flop
        return features

    @staticmethod
    def _calculate_straight_potential(sorted_ranks):
        """Efficient straight draw potential calculation"""
        if len(sorted_ranks) < 3:
            return 0.0
            
        # Count consecutive ranks
        gaps = np.diff(sorted_ranks)
        consecutive = np.sum(gaps == 1) + 1
        
        # Check if we have 4 consecutive cards
        if consecutive >= 4:
            return 1.0
            
        # Check for Ace-low straight potential
        if 14 in sorted_ranks and {2, 3, 4, 5} & set(sorted_ranks):
            return 1.0
            
        # Check if we have 3 consecutive with gaps that could complete straight
        if consecutive >= 3:
            gaps_fillable = np.sum(gaps <= 2)
            if gaps_fillable >= 3:
                return 1.0
                
        return 0.0

    @staticmethod
    def _calculate_full_house_potential(rank_counts):
        """Efficient full house draw potential calculation"""
        pairs = np.sum(rank_counts == 2)
        three_kind = np.sum(rank_counts == 3)
        
        # Already have full house
        if (three_kind >= 1 and pairs >= 1) or three_kind >= 2:
            return 0.0
            
        # Full house draw potentials:
        if three_kind >= 1:  # Need one pair
            return 1.0
        if pairs >= 2:  # Can improve one pair to trips
            return 1.0
            
        return 0.0

    @staticmethod
    def _high_card_strength(hole_cards):
        """Calculate relative strength of high card (Ace high = 1.0, 2 high = 0.0)"""
        max_rank = max(Card.get_rank_int(card) for card in hole_cards)
        return (max_rank - 2) / 12  # Normalize to 0-1

    @staticmethod
    def _pair_strength(ranks, hole_cards):
        """Calculate relative strength of pair"""
        hole_ranks = [Card.get_rank_int(card) for card in hole_cards]
        pair_rank = max(r for r in ranks if ranks.count(r) >= 2)
        
        # Premium pair (pocket pair)
        if hole_ranks[0] == hole_ranks[1] == pair_rank:
            return (pair_rank - 2) / 12 + 0.5  # Bonus for pocket pairs
        
        # Top pair vs middle pair vs bottom pair relative to board
        unique_ranks = sorted(set(ranks))
        rank_position = unique_ranks.index(pair_rank)
        return rank_position / len(unique_ranks)

    @staticmethod
    def _two_pair_strength(ranks, hole_cards):
        """Calculate relative strength of two pair"""
        hole_ranks = [Card.get_rank_int(card) for card in hole_cards]
        pairs = sorted([r for r in ranks if ranks.count(r) >= 2], reverse=True)
        
        # Both pairs in hole cards
        if all(r in hole_ranks for r in pairs[:2]):
            return 1.0
            
        # One pair in hole cards
        if any(r in hole_ranks for r in pairs[:2]):
            return 0.5 + (max(pairs) - 2) / 24
            
        return max(pairs) / 14

    @staticmethod
    def _trips_strength(ranks, hole_cards):
        """Calculate relative strength of three of a kind"""
        hole_ranks = [Card.get_rank_int(card) for card in hole_cards]
        trips_rank = next(r for r in ranks if ranks.count(r) >= 3)
        
        # Set in the hole
        if hole_ranks[0] == hole_ranks[1] == trips_rank:
            return 1.0
        
        # One card contributing to trips
        if trips_rank in hole_ranks:
            return 0.7 + (trips_rank - 2) / 40
            
        return (trips_rank - 2) / 12

    @staticmethod
    def _straight_strength(ranks):
        """
        Calculate relative strength of a straight.

        Args:
            ranks (list): List of rank integers.

        Returns:
            float: Relative strength of the straight (0.0 to 1.0).
        """
        sorted_ranks = sorted(set(ranks))  # Unique and sorted ranks
        if len(sorted_ranks) < 5:
            return 0.0  # Not enough cards to form a straight

        # Check for standard straights
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i:i + 5] == list(range(sorted_ranks[i], sorted_ranks[i] + 5)):
                return (sorted_ranks[i + 4] - 5) / 9  # Normalize high card to 0-1

        # Check for Ace-low straight (A, 2, 3, 4, 5)
        if {14, 2, 3, 4, 5}.issubset(set(ranks)):
            return 0.0  # Special case for the lowest straight

        return 0.0  # No straight found


    @staticmethod
    def _flush_strength(cards):
        """Calculate relative strength of flush"""
        suits = [Card.get_suit_int(card) for card in cards]
        flush_suit = max(set(suits), key=suits.count)
        flush_cards = [c for c in cards if Card.get_suit_int(c) == flush_suit]
        high_rank = max(Card.get_rank_int(c) for c in flush_cards)
        return (high_rank - 2) / 12

    @staticmethod
    def _full_house_strength(ranks, hole_cards):
        """Calculate relative strength of full house"""
        hole_ranks = [Card.get_rank_int(card) for card in hole_cards]
        trips_rank = next(r for r in ranks if ranks.count(r) >= 3)
        pair_rank = next(r for r in ranks if ranks.count(r) >= 2 and r != trips_rank)
        
        # Both hole cards part of trips
        if sum(1 for r in hole_ranks if r == trips_rank) == 2:
            return 1.0
        
        # Both components use hole cards
        if any(r == trips_rank for r in hole_ranks) and any(r == pair_rank for r in hole_ranks):
            return 0.8
            
        return (trips_rank - 2) / 12

    @staticmethod
    def _four_kind_strength(ranks, hole_cards):
        """Calculate relative strength of four of a kind"""
        hole_ranks = [Card.get_rank_int(card) for card in hole_cards]
        quads_rank = next(r for r in ranks if ranks.count(r) >= 4)
        
        # Both hole cards part of quads
        if sum(1 for r in hole_ranks if r == quads_rank) == 2:
            return 1.0
            
        # One hole card part of quads
        if any(r == quads_rank for r in hole_ranks):
            return 0.8
            
        return (quads_rank - 2) / 12
    
    @staticmethod
    def _calculate_draw_potentials(player_hand, community_cards):
        """
        Calculate draw potentials for flush, straight, and full house.

        Args:
            player_hand (list): Two cards representing the player's hand.
            community_cards (list): Community cards on the board.

        Returns:
            np.array: [flush_potential, straight_potential, full_house_potential]
        """
        all_cards = player_hand + community_cards
        ranks = [Card.get_rank_int(card) for card in all_cards]
        suits = [Card.get_suit_int(card) for card in all_cards]

        # Flush potential
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        flush_potential = max(suit_counts.values()) / 7  # Normalize by total cards

        # Straight potential
        sorted_ranks = sorted(set(ranks))
        straight_potential = HandEvaluator._calculate_straight_potential(sorted_ranks)

        # Full house potential
        rank_counts = np.array([ranks.count(rank) for rank in set(ranks)])
        full_house_potential = HandEvaluator._calculate_full_house_potential(rank_counts)

        return np.array([flush_potential, straight_potential, full_house_potential], dtype=np.float32)