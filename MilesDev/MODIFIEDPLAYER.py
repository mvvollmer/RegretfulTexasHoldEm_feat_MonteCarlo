from pokerenv.common import PlayerAction
from pokerenv.player import Player


class ModifiedPlayer(Player):
    def __init__(self, identifier, name, penalty):
        super().__init__(identifier, name, penalty)
        
    def call(self, amount_to_call):
        self.has_acted = True
        self.acted_this_street = True

        # Determine actual call size based on stack and amount to call
        call_size = min(amount_to_call, self.stack)
        self.stack -= call_size
        self.bet_this_street += call_size
        self.money_in_pot += call_size

        if self.stack == 0:
            self.all_in = True

        self.history.append({'action': PlayerAction.CALL, 'value': call_size})
        return call_size

    def bet(self, amount):
        self.has_acted = True
        self.acted_this_street = True

        # Ensure valid bet amount
        if amount > self.stack:
            amount = self.stack

        # Place the bet
        bet_size = amount
        self.stack -= bet_size
        self.bet_this_street += bet_size
        self.money_in_pot += bet_size

        if self.stack == 0:
            self.all_in = True

        self.history.append({'action': PlayerAction.BET, 'value': bet_size})
        return bet_size
