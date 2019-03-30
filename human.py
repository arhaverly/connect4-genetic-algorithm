class HumanPlayer():
    def get_move(self):
        move = -1
        try:
            move = int(input())
        except ValueError:
            pass
        return move