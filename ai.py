import random
class AIPlayer():

    def __init__(self, session, Y_index, Y_, W1, B1, W2, B2, X, generation):

        self.session = session
        self.Y_index = Y_index
        self.Y_ = Y_
        self.W1 = W1
        self.B1 = B1
        self.W2 = W2
        self.B2 = B2
        self.X = X
        self.epsilon = 10000 - generation

    def get_move(self, game):
        inputs = [col/2 for row in game.board for col in row]
        if not game.player1_turn:
            for item in inputs:
                if item == 0.5:
                    item = 1
                if item == 1:
                    item = 0.5
        
        low_chance = random.randint(0, 100)


        if random.randint(0, 20000) < self.epsilon or low_chance < 5:
            move = random.randint(0, 6)
            random_move = True
            # print('random: ', end='')
        else:
            move, p, w1, b1, w2, b2 = self.session.run([self.Y_index, self.Y_, self.W1, self.B1, self.W2, self.B2], feed_dict={self.X:[inputs]})
            random_move = False
            # print('prediction: ', end='')

        # move = int(move)
        return int(move), random_move

