import pokerenv.obs_indices as indices
from pokerenv.table import Table
from treys import Deck, Evaluator, Card
from pokerenv.common import GameState, PlayerState, PlayerAction, TablePosition, Action, action_list
from pokerenv.player import Player
from pokerenv.utils import pretty_print_hand, approx_gt, approx_lte
import numpy as np
from MODIFIEDPLAYER import ModifiedPlayer

class ModifiedTable(Table):
    def __init__(self, n_players, player_names=None, track_single_player=False, stack_low=50, stack_high=200, hand_history_location='hands/', invalid_action_penalty=0, SB=5, BB=10):
        super().__init__(n_players, player_names, track_single_player, stack_low, stack_high, hand_history_location, invalid_action_penalty)
        self.current_dealer = 0
        self.side_pots = []
        self.SB = SB
        self.BB = BB
        if player_names is None:
            player_names = {}
        for player in range(6):
            if player not in player_names.keys():
                player_names[player] = 'player_%d' % (player+1)
        self.all_players = [ModifiedPlayer(n, player_names[n], invalid_action_penalty) for n in range(6)]
        self.players = self.all_players[:n_players]
        self.OriginalPlayers = self.players
        self.OriginalPlayer_names = player_names
        self.OriginalN_players = n_players
        
    def reset(self):
        self.hands = 0
        self.current_turn = 0
        self.pot = 0
        self.street = GameState.PREFLOP
        self.deck.cards = Deck.GetFullDeck()
        self.rng.shuffle(self.deck.cards)
        self.cards = []
        self.players = self.OriginalPlayers
        self.player_names = self.OriginalPlayer_names
        self.n_players = self.OriginalN_players
        self.street_finished = False
        self.hand_is_over = False
        
        # Deal initial cards
        initial_draw = self.deck.draw(self.n_players * 2)
        for i, player in enumerate(self.players):
            player.reset()
            player.cards = [initial_draw[i], initial_draw[i + self.n_players]]
            player.stack = self.rng.integers(self.stack_low, self.stack_high, 1)[0]

        # Rotate dealer position
        self.current_dealer = (self.current_dealer + 1) % self.n_players
        sb_position = (self.current_dealer + 1) % self.n_players
        bb_position = (sb_position + 1) % self.n_players if self.n_players > 2 else self.current_dealer
       
        # Assign blinds
        self.pot += self.players[sb_position].bet(self.SB)
        self._write_event(f"{self.players[sb_position].name}: posts small blind ${self.SB:.2f}, Player money in pot: ${self.players[sb_position].money_in_pot:.2f}, Player stack: ${self.players[sb_position].stack:.2f}")
        
        self.pot += self.players[bb_position].bet(self.BB)
        self._change_bet_to_match(self.BB)  # Update bet-to-match for the big blind only
        self.last_bet_placed_by = self.players[bb_position]  # Set the last bet placed by the big blind
        self._write_event(f"{self.players[bb_position].name}: posts big blind ${self.BB:.2f}, Player money in pot: ${self.players[bb_position].money_in_pot:.2f}, Player stack: ${self.players[bb_position].stack:.2f}")

        

        # Determine the first player to act
        self.first_to_act = (bb_position + 1) % self.n_players if self.n_players > 2 else sb_position
        self.next_player_i = self.first_to_act

        return self._get_observation(self.players[self.next_player_i])
    
    def resetHand(self, info):
        self.current_turn = 0
        self.pot = 0
        self.street = GameState.PREFLOP
        self.deck.cards = Deck.GetFullDeck()
        self.rng.shuffle(self.deck.cards)
        self.cards = []
        self.players = []
        self.player_names = []
               
        self.street_finished = False
        self.hand_is_over = False
        
        for player, name in info:
            self.players.append(player)
            self.player_names.append(name)
        self.n_players = len(self.players)
            
        # Deal initial cards
        initial_draw = self.deck.draw(self.n_players * 2)
        for i, player in enumerate(self.players):
            player.reset()
            player.cards = [initial_draw[i], initial_draw[i + self.n_players]]
            
        
        # Current code
        self.current_dealer = (self.current_dealer + 1) % self.n_players
        sb_position = (self.current_dealer + 1) % self.n_players
        bb_position = (sb_position + 1) % self.n_players if self.n_players > 2 else self.current_dealer

        # Consider tracking active players for position calculation
        active_players = [i for i in range(self.n_players) if self.players[i].stack > 0]
        if len(active_players) < 2:
            raise ValueError("Not enough players with chips to continue")


        # Assign blinds
        self.pot += self.players[sb_position].bet(self.SB)
        self._write_event(f"{self.players[sb_position].name}: posts small blind ${self.SB:.2f}, Player money in pot: ${self.players[sb_position].money_in_pot:.2f}, Player stack: ${self.players[sb_position].stack:.2f}")
        
        self.pot += self.players[bb_position].bet(self.BB)
        self._change_bet_to_match(self.BB)  # Update bet-to-match for the big blind only
        self.last_bet_placed_by = self.players[bb_position]  # Set the last bet placed by the big blind
        self._write_event(f"{self.players[bb_position].name}: posts big blind ${self.BB:.2f}, Player money in pot: ${self.players[bb_position].money_in_pot:.2f}, Player stack: ${self.players[bb_position].stack:.2f}")
        
        # Determine the first player to act
        self.first_to_act = (bb_position + 1) % self.n_players if self.n_players > 2 else sb_position
        self.next_player_i = self.first_to_act
     
    
        return self._get_observation(self.players[self.next_player_i])
    
        
    def _int_to_str(card_int: int) -> str:
        rank_int = Card.get_rank_int(card_int)
        suit_int = Card.get_suit_int(card_int)
        return Card.STR_RANKS[rank_int] + Card.INT_SUIT_TO_CHAR_SUIT[suit_int]
    
    def _get_rank_int(card_int: int) -> int:
        if card_int is int:
            return (card_int >> 8) & 0xF
        else:
            return (card_int[0] >> 8) & 0xF
        
    def _get_suit_int(card_int: int) -> int:
        if card_int is int:
            return (card_int >> 12) & 0xF
        else:
            return (card_int[0] >> 12) & 0xF
        
    def _get_bitrank_int(card_int: int) -> int:
        if card_int is int:
            return (card_int >> 16) & 0x1FFF
        else:
            return (card_int[0] >> 16) & 0x1FFF

    def _get_prime(card_int: int) -> int:
        if card_int is int:
            return card_int & 0x3F
        else:
            return card_int[0] & 0x3F

    def _street_transition(self, transition_to_end=False):
            transitioned = False
            if self.street == GameState.PREFLOP:
                self.cards = self.deck.draw(3)
                self._write_event("*** FLOP *** [%s %s %s]" %
                                (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                Card.int_to_str(self.cards[2])))
                self.street = GameState.FLOP
                transitioned = True
            if self.street == GameState.FLOP and (not transitioned or transition_to_end):
                new = self.deck.draw(1)[0]
                self.cards.append(new)
                self._write_event("*** TURN *** [%s %s %s] [%s]" %
                                (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3])))
                self.street = GameState.TURN
                transitioned = True
            if self.street == GameState.TURN and (not transitioned or transition_to_end):
                new = self.deck.draw(1)[0]
                self.cards.append(new)
                self._write_event("*** RIVER *** [%s %s %s %s] [%s]" %
                                (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3]),
                                Card.int_to_str(self.cards[4])))
                self.street = GameState.RIVER
                transitioned = True
            if self.street == GameState.RIVER and (not transitioned or transition_to_end):
                if not self.hand_is_over:
                    if self.hand_history_enabled:
                        self._write_show_down()
                self.hand_is_over = True
            self.street_finished = False
            self.last_bet_placed_by = None
            self.first_to_act = None
            self.bet_to_match = 0
            self.minimum_raise = 0
            for player in self.players:
                player.finish_street()

    def step(self, action: Action):
        BB = self.BB
        self.current_player_i = self.next_player_i
        player = self.players[self.current_player_i]
        self.current_turn += 1
        
        #self.debug_pot_distribution()
        
        # Skip invalid player states
        if (player.all_in or player.state is not PlayerState.ACTIVE) and not self.hand_is_over:
            # Instead of raising an exception, find next valid player
            active_players_after = [i for i in range(self.n_players) 
                                if i > self.current_player_i 
                                if self.players[i].state is PlayerState.ACTIVE 
                                if not self.players[i].all_in]
            active_players_before = [i for i in range(self.n_players) 
                                if i <= self.current_player_i 
                                if self.players[i].state is PlayerState.ACTIVE 
                                if not self.players[i].all_in]
            
            if len(active_players_after) > 0:
                self.next_player_i = min(active_players_after)
            elif len(active_players_before) > 0:
                self.next_player_i = min(active_players_before)
            else:
                # If no active players, end the hand
                self.hand_is_over = True
                
            obs = np.zeros(self.observation_space.shape[0]) if self.hand_is_over else self._get_observation(self.players[self.next_player_i])
            rewards = np.asarray([player.get_reward() for player in sorted(self.players)])
            return obs, rewards, self.hand_is_over, {}

        if self.first_to_act is None:
            self.first_to_act = player

        # Apply the player action
        if not (self.hand_is_over or self.street_finished):
            valid_actions = self._get_valid_actions(player)
            if not self._is_action_valid(player, action, valid_actions):
                player.punish_invalid_action()
            elif action.action_type is PlayerAction.FOLD:
                if self.bet_to_match > 0:
                    player.fold()
                    self.active_players -= 1
                else:
                    player.check()
                    player.state = PlayerState.ACTIVE
                self._write_event(f"{player.name}: folds, Player money in pot: ${player.money_in_pot:.2f}, Player stack: ${player.stack}")
            elif action.action_type is PlayerAction.CHECK:
                player.check()
                self._write_event(f"{player.name}: checks, Player money in pot: ${player.money_in_pot:.2f}, Player stack: ${player.stack}")
            elif action.action_type is PlayerAction.CALL:
                call_size = player.call(self.bet_to_match)
                self.pot += call_size
                if player.all_in:
                    self._write_event(f"{player.name}: calls ${call_size} and is all-in, Player money in pot: ${player.money_in_pot:.2f}, Player stack: ${player.stack:.2f}")
                else:
                    self._write_event(f"{player.name}: calls ${call_size}, Player money in pot: ${player.money_in_pot:.2f}, Player stack: ${player.stack:.2f}")
            elif action.action_type is PlayerAction.BET:
                previous_bet_this_street = player.bet_this_street
                actual_bet_size = player.bet(np.round(action.bet_amount, 2))
                self.pot += actual_bet_size
                if self.bet_to_match == 0:
                    if player.all_in:
                        self._write_event(f"{player.name}: bets ${actual_bet_size} and is all-in, Player money in pot: ${player.money_in_pot:.2f}, Player stack: ${player.stack:.2f}")
                    else:
                        self._write_event(f"{player.name}: bets ${actual_bet_size}, Player money in pot: ${player.money_in_pot:.2f}, Player stack: ${player.stack:.2f}")
                else:
                    if player.all_in:
                        self._write_event("%s: raises $%.2f to $%.2f and is all-in" %
                                            (player.name,
                                            ((actual_bet_size + previous_bet_this_street) - self.bet_to_match),
                                            (actual_bet_size + previous_bet_this_street))
                                            )
                    else:
                        self._write_event("%s: raises $%.2f to $%.2f" %
                                            (player.name,
                                            ((actual_bet_size + previous_bet_this_street) - self.bet_to_match),
                                            (actual_bet_size + previous_bet_this_street))
                                            )
                self._change_bet_to_match(actual_bet_size + previous_bet_this_street)
                self.last_bet_placed_by = player
            else:
                raise Exception("Error when parsing action, make sure player action_type is PlayerAction and not int")

            should_transition_to_end = False
            players_with_actions = [p for p in self.players if p.state is PlayerState.ACTIVE if not p.all_in]
            players_who_should_act = [p for p in players_with_actions if (not p.acted_this_street or p.bet_this_street != self.bet_to_match)]

            # If the game is over, or the betting street is finished, progress the game state
            if len(players_with_actions) < 2 and len(players_who_should_act) == 0:
                # Calculate side pots when we have all-in situations
                if self.active_players > 1:
                    # Get all active players and their bets, sorted by bet size
                    active_bets = [(p, p.bet_this_street) for p in self.players if p.state is PlayerState.ACTIVE]
                    active_bets.sort(key=lambda x: x[1])
                    
                    # Clear existing side pots
                    self.side_pots = []
                    remaining_players = len(active_bets)
                    
                    # Calculate side pots
                    for i in range(len(active_bets) - 1):
                        current_bet = active_bets[i][1]
                        next_bet = active_bets[i + 1][1]
                        
                        if current_bet < next_bet:
                            # Create side pot
                            side_pot_amount = (next_bet - current_bet) * (remaining_players - i - 1)
                            if side_pot_amount > 0:
                                self.side_pots.append({
                                    'amount': side_pot_amount,
                                    'eligible_players': [p[0] for p in active_bets[i+1:]],
                                    'level': next_bet
                                })

                amount = 0
                # If all active players are all-in, transition to the end
                if self.active_players > 1:
                    active_bettors = [p.bet_this_street for p in self.players 
                                    if p.state is PlayerState.ACTIVE 
                                    if p is not self.last_bet_placed_by]
                    
                    if active_bettors:  # Check if there are any other active players
                        biggest_bet_call = max(active_bettors)
                        last_bet_this_street = 0
                        if self.last_bet_placed_by is not None:
                            last_bet_this_street = self.last_bet_placed_by.bet_this_street
                        if biggest_bet_call < last_bet_this_street:
                            amount = last_bet_this_street - biggest_bet_call
                        should_transition_to_end = True
                    else:
                        # If no other active bettors, use the last bet as the amount
                        amount = self.minimum_raise if self.minimum_raise > 0 else self.bet_to_match
                        should_transition_to_end = True
                else:
                    # If only one active player, end the hand
                    self.hand_is_over = True
                    amount = self.minimum_raise if self.minimum_raise > 0 else self.bet_to_match

                # If there are uncalled bets, return them to the player who placed them
                if amount > 0:
                    self.pot -= amount
                    if self.last_bet_placed_by is not None:
                        self.last_bet_placed_by.stack += amount
                        self.last_bet_placed_by.money_in_pot -= amount
                        self.last_bet_placed_by.bet_this_street -= amount
                        self._write_event(
                            f"Uncalled bet (${amount}) returned to {self.last_bet_placed_by.name}"
                        )
                    else:
                        print(f"Warning: No last_bet_placed_by to return ${amount}")
                if should_transition_to_end:
                    self._street_transition(transition_to_end=True)

            # If the betting street is still active, choose next player to act
            else:
                active_players_after = [i for i in range(self.n_players) if i > self.current_player_i if
                                        self.players[i].state is PlayerState.ACTIVE if not self.players[i].all_in]
                active_players_before = [i for i in range(self.n_players) if i <= self.current_player_i if
                                         self.players[i].state is PlayerState.ACTIVE if not self.players[i].all_in]
                if len(active_players_after) > 0:
                    self.next_player_i = min(active_players_after)
                else:
                    self.next_player_i = min(active_players_before)
                next_player = self.players[self.next_player_i]
                if self.last_bet_placed_by is next_player or (self.first_to_act is next_player and self.last_bet_placed_by is None):
                    self.street_finished = True
                    if len(active_players_before) > 0:
                        self.next_player_i = min(active_players_before)

        if self.street_finished and not self.hand_is_over:
            self._street_transition()

        obs = np.zeros(self.observation_space.shape[0]) if self.hand_is_over else self._get_observation(self.players[self.next_player_i])
        rewards = np.asarray([player.get_reward() for player in sorted(self.players)])
        if self.hand_is_over:
            self._distribute_pot()
            self._finish_hand()
            
        #self.debug_pot_distribution()
            
        return obs, rewards, self.hand_is_over, {}
    
    def _distribute_pot(self):
        """
        Distributes the pot and calculates profit/loss for each player.
        """
        # Validate pot total
        total_contributions = sum(p.money_in_pot for p in self.players)
        
        if abs(self.pot - total_contributions) > 0.01:
            self.pot = total_contributions

        active_players = [p for p in self.players if p.state is PlayerState.ACTIVE]
        
        # Initialize winnings values
        for player in self.players:
            player.winnings = -player.money_in_pot
            player.winnings_for_hh = 0  # Don't initialize this negative

        # Single player case
        if len(active_players) == 1:
            winner = active_players[0]
            winner.winnings = self.pot - winner.money_in_pot
            winner.winnings_for_hh = self.pot
            winner.stack += self.pot
            self._write_event(f"{winner.name} won ${winner.winnings:.2f}")
            return
        
           # Preflop all-in scenario
        if len(self.cards) == 0:
            # Helper function to get card rank (2-14 where 14 is Ace)
            def get_card_rank(card):
                rank_int = Card.get_rank_int(card)
                # Convert rank to 2-14 scale where Ace is 14
                ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                        '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                return ranks[Card.STR_RANKS[rank_int]]

            # Helper function to evaluate preflop hand strength
            def evaluate_hole_cards(cards):
                rank1 = get_card_rank(cards[0])
                rank2 = get_card_rank(cards[1])
                suit1 = Card.get_suit_int(cards[0])
                suit2 = Card.get_suit_int(cards[1])
                
                # Score based on:
                # 1. High card value (max of two cards)
                # 2. Pair value (if pair exists)
                # 3. Suited bonus
                # 4. Second card value
                
                is_pair = rank1 == rank2
                is_suited = suit1 == suit2
                high_card = max(rank1, rank2)
                low_card = min(rank1, rank2)
                
                if is_pair:
                    score = rank1 * 1000000  # Pairs are always better than non-pairs
                else:
                    score = (high_card * 10000) + (low_card * 100) + (100 if is_suited else 0)
                
                return score

            # Calculate hand ranks for preflop
            for player in active_players:
                player.hand_rank = evaluate_hole_cards(player.cards)

        else:
            # Calculate hand ranks using community cards
            for player in active_players:
                player.calculate_hand_rank(self.evaluator, self.cards)
                
        # Sort by contribution amount
        active_players.sort(key=lambda x: x.money_in_pot)

        # Process each pot level
        prev_contribution = 0
        for i, current_player in enumerate(active_players):
            current_contribution = current_player.money_in_pot
            if current_contribution > prev_contribution:
                pot_size = (current_contribution - prev_contribution) * (len(active_players) - i)
                eligible_players = active_players[i:]
                
                best_hand = min(p.hand_rank for p in eligible_players)
                winners = [p for p in eligible_players if p.hand_rank == best_hand]
                
                share = pot_size / len(winners)
                for winner in winners:
                    winner.winnings += share
                    winner.winnings_for_hh += share
                    winner.stack += share

            prev_contribution = current_contribution

        # Write final profit/loss messages
        for player in active_players:
            if player.winnings > 0:
                self._write_event(f"{player.name} won ${player.winnings:.2f}")
            else:
                self._write_event(f"{player.name} lost ${-player.winnings:.2f}")
                
                
                
    def debug_pot_distribution(self):
        """
        Debug pot distribution to ensure stacks and pot calculations are consistent.
        """
        print("---- Debugging Pot Distribution ----")
        print(f"Total Pot: ${self.pot:.2f}")
        print("Player Contributions to Pot:")
        for player in self.players:
            print(f"  {player.name}: Money in Pot: ${player.money_in_pot:.2f}, Stack: ${player.stack:.2f}")

        # Validate total contributions match pot
        total_contributions = sum(player.money_in_pot for player in self.players)
        if abs(self.pot - total_contributions) > 0.01:
            print(f"ERROR: Pot (${self.pot:.2f}) does not match total contributions (${total_contributions:.2f}).")
        else:
            print("Pot matches total contributions.")

        # Check for negative stack or money_in_pot
        for player in self.players:
            if player.stack < 0:
                print(f"ERROR: Player {player.name} has a negative stack: ${player.stack:.2f}.")
            if player.money_in_pot < 0:
                print(f"ERROR: Player {player.name} has negative money in pot: ${player.money_in_pot:.2f}.")

        # Check side pot calculations
        if self.side_pots:
            print("Side Pot Details:")
            for i, side_pot in enumerate(self.side_pots):
                eligible_player_names = [p.name for p in side_pot['eligible_players']]
                print(f"  Side Pot {i+1}: Amount: ${side_pot['amount']:.2f}, Level: ${side_pot['level']:.2f}")
                print(f"    Eligible Players: {eligible_player_names}")

        print("---- End Debugging Pot Distribution ----")
