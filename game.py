import numpy

class Game():

    def __init__(self):
        self.board = numpy.zeros((6, 7)) #[[0 for _ in range(7)] for _ in range(6)]
        self.player1_turn = True

    def print_board(self, file):
        if self.player1_turn == True:
            file.write('player1\'s turn\n')
        else:
            file.write('player2\'s turn\n')

        for row in self.board:
            for col in row:
                file.write(str(int(col)) + '  ')

            file.write('\n')

        file.write('-------------------\n')
        for i in range(7):
            file.write(str(i) + '  ')

        file.write('\n')

    def set_board(self, board):
        self.board = board

    def game_over(self):
        return self.vertical_win() or self.horizontal_win() or self.diagonal_win() or self.board_full()

    def is_valid_move(self, move):
        if move < 0 or move > 6:
            return False

        if self.board[0][move] == 0:
            return True

        return False

    def add_move(self, move):
        if self.player1_turn:
            piece = 1
        else:
            piece = 2
        for row in self.board[::-1]:
            if row[move] == 0:
                row[move] = piece
                return

    def update_counters(self, i, j, counter1, counter2):
        if self.board[i][j] == 0:
            counter1 = 0
            counter2 = 0
        elif self.board[i][j] == 1:
            counter1 += 1
            counter2 = 0
        else:
            counter1 = 0
            counter2 += 1

        return counter1, counter2

    def vertical_win(self):
        for j in range(len(self.board[0])):
            counter1 = 0
            counter2 = 0
            for i in range(len(self.board)):
                counter1, counter2 = self.update_counters(i, j, counter1, counter2)
                if counter1 > 3 or counter2 > 3:
                    return True
        return False

    def horizontal_win(self):
        for i in range(len(self.board)):
            counter1 = 0
            counter2 = 0
            for j in range(len(self.board[0])):
                counter1, counter2 = self.update_counters(i, j, counter1, counter2)
                if counter1 > 3 or counter2 > 3:
                    return True
        return False

    def diagonal_win(self):
        return self.diagonal_win_right() or self.diagonal_win_left()

    def diagonal_win_right(self):
        starting_points = [[2,0], [1,0], [0,0], [0,1], [0,2], [0,3]]
        for i, j in starting_points:
            counter1 = 0
            counter2 = 0
            while i < 6 and j < 7:
                counter1, counter2 = self.update_counters(i, j, counter1, counter2)
                if counter1 > 3 or counter2 > 3:
                    return True
                i += 1
                j += 1

        return False

    def diagonal_win_left(self):
        starting_points = [[2,6], [1,6], [0,6], [0,5], [0,4], [0,3]]
        for i, j in starting_points:
            counter1 = 0
            counter2 = 0
            while i < 6 and j > -1:
                counter1, counter2 = self.update_counters(i, j, counter1, counter2)
                if counter1 > 3 or counter2 > 3:
                    return True
                i += 1
                j -= 1

        return False

    def board_full(self):
        for col in self.board[0]:
            if col == 0:
                return False

        return True