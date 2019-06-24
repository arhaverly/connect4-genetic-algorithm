import random
class AIPlayer():

    def __init__(self, session, Y_index, Y_, W1, B1, W2, B2, W3, B3, W4, B4, X, generation):

        self.session = session
        self.Y_index = Y_index
        self.Y_ = Y_
        self.W1 = W1
        self.B1 = B1
        self.W2 = W2
        self.B2 = B2
        self.W3 = W3
        self.B3 = B3
        self.W4 = W4
        self.B4 = B4
        self.X = X

    def get_move(self, game):
        inputs = []
        for row in game.board:
            for col in row:
                print(col)
        
        # inputs = [col/2 for row in game.board for col in row]
        if not game.player1_turn:
            for item in inputs:
                if item == 0.5:
                    item = 1
                if item == 1:
                    item = 0.5

        move, p, w1, b1, w2, b2, w3, b3, w4, b4 = self.session.run([self.Y_index, self.Y_, self.W1, self.B1, self.W2, self.B2, self.W3, self.B3, self.W4, self.B4], feed_dict={self.X:[inputs]})


        # move = int(move)
        return int(move)

