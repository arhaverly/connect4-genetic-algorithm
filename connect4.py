import random
from game import *
import human
import ai

import tensorflow as tf
import numpy as np
import argparse

RECORD_GEN = 20000

def mutate_w_with_percent_change(p, add_sub_rand=True):
    #considering its 2d array
    new_p = []
    for i in p:
        row = []
        for j in i:
            temp = j
            delta = np.random.random_sample() + 0.5
            if np.random.random_sample() > 0.5:
                temp = temp * delta
            if add_sub_rand == True:
                if np.random.random_sample() > 0.5:
                    if np.random.random_sample() > 0.5:
                        temp = temp - np.random.random_sample()
                    else:
                        temp = temp + np.random.random_sample()
            row.append(temp)
        new_p.append(row)
    return new_p

def mutate_b_with_percent_change(p, add_sub_rand=True):
    #considering its 1d array
    new_p = []
    for i in p:
        temp = i
        delta = np.random.random_sample() + 0.5
        if np.random.random_sample() > 0.5:
            temp = temp * delta
        if add_sub_rand == True:
            if np.random.random_sample() > 0.5:
                if np.random.random_sample() > 0.5:
                    temp = temp - np.random.random_sample()
                else:
                    temp = temp + np.random.random_sample()
        new_p.append(temp)
    return new_p

def cross_over(w11, w12, w13, w14, b11, b12, b13, b14, w21, w22, w23, w24, b21, b22, b23, b24):
    new_w1 = []
    for i in range(len(w11)):
        row = []
        for j in range(len(w11[0])):
            if np.random.random_sample() > 0.5:
                row.append(w11[i][j])
            else:
                row.append(w21[i][j])
        new_w1.append(row)

    new_w2 = []
    for i in range(len(w12)):
        row = []
        for j in range(len(w12[0])):
            if np.random.random_sample() > 0.5:
                row.append(w12[i][j])
            else:
                row.append(w22[i][j])
        new_w2.append(row)

    new_w3 = []
    for i in range(len(w13)):
        row = []
        for j in range(len(w13[0])):
            if np.random.random_sample() > 0.5:
                row.append(w13[i][j])
            else:
                row.append(w23[i][j])
        new_w3.append(row)

    new_w4 = []
    for i in range(len(w14)):
        row = []
        for j in range(len(w14[0])):
            if np.random.random_sample() > 0.5:
                row.append(w14[i][j])
            else:
                row.append(w24[i][j])
        new_w4.append(row)
    
    new_b1 = []
    for i in range(len(b11)):
        if np.random.random_sample() > 0.5:
            new_b1.append(b11[i])
        else:
            new_b1.append(b21[i])

    new_b2 = []
    for i in range(len(b12)):
        if np.random.random_sample() > 0.5:
            new_b2.append(b12[i])
        else:
            new_b2.append(b22[i])

    new_b3 = []
    for i in range(len(b13)):
        if np.random.random_sample() > 0.5:
            new_b3.append(b13[i])
        else:
            new_b3.append(b23[i])

    new_b4 = []
    for i in range(len(b14)):
        if np.random.random_sample() > 0.5:
            new_b4.append(b14[i])
        else:
            new_b4.append(b24[i])

    return (new_w1, new_w2, new_w3, new_w4, new_b1, new_b2, new_b3, new_b4)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', nargs='?')

    args = parser.parse_args()

    POPULATION_SIZE = 40
    MUTATION_PROBABILITY = 0.8
    W_MUTATION_PROBABILITY = 0.5
    B_MUTATION_PROBABILITY = 0.5
    MAX_GEN = 10000
    N_EPISODE = 10
    is_training_finished = False

    graph = tf.Graph()
    with graph.as_default():
        num_input = 42
        hidden_units0 = 200
        hidden_units1 = 100
        hidden_units2 = 50
        num_class = 7

        X = tf.placeholder(tf.float32, shape=[1, num_input], name='X')

        W1 = tf.Variable(tf.random_normal([num_input, hidden_units0], stddev=1.0), name='W1')
        B1 = tf.Variable(tf.random_normal([hidden_units0], stddev=1.0) , name='B1')
        A1 = tf.nn.softmax(tf.matmul(X, W1) + B1,  name='A1')

        W2 = tf.Variable(tf.random_normal([hidden_units0, hidden_units1], stddev=1.0), name='W2')
        B2 = tf.Variable(tf.random_normal([hidden_units1], stddev=1.0) , name='B2')
        A2 = tf.nn.softmax(tf.matmul(A1, W2) + B2,  name='A2')

        W3 = tf.Variable(tf.random_normal([hidden_units1, hidden_units2], stddev=1.0), name='W3')
        B3 = tf.Variable(tf.random_normal([hidden_units2], stddev=1.0) , name='B3')
        A3 = tf.nn.softmax(tf.matmul(A2, W3) + B3,  name='A3')

        W4 = tf.Variable(tf.random_normal([hidden_units2, num_class], stddev=1.0), name='W4')
        B4 = tf.Variable(tf.random_normal([num_class], stddev=1.0), name='B4')
        Y_ = tf.nn.softmax(tf.matmul(A3, W4) + B4, name='Y')
        Y_index = tf.argmax(Y_,1)

        W1_placeholder = tf.placeholder(tf.float32, shape=[num_input, hidden_units0])
        W2_placeholder = tf.placeholder(tf.float32, shape=[hidden_units0, hidden_units1])
        W3_placeholder = tf.placeholder(tf.float32, shape=[hidden_units1, hidden_units2])
        W4_placeholder = tf.placeholder(tf.float32, shape=[hidden_units2, num_class])

        W1_assign = tf.assign(W1, W1_placeholder)
        W2_assign = tf.assign(W2, W2_placeholder)
        W3_assign = tf.assign(W3, W3_placeholder)
        W4_assign = tf.assign(W4, W4_placeholder)

        B1_placeholder = tf.placeholder(tf.float32, shape=[hidden_units0])
        B2_placeholder = tf.placeholder(tf.float32, shape=[hidden_units1])
        B3_placeholder = tf.placeholder(tf.float32, shape=[hidden_units2])
        B4_placeholder = tf.placeholder(tf.float32, shape=[num_class])

        B1_assign = tf.assign(B1, B1_placeholder)
        B2_assign = tf.assign(B2, B2_placeholder)
        B3_assign = tf.assign(B3, B3_placeholder)
        B4_assign = tf.assign(B4, B4_placeholder)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=POPULATION_SIZE)

    if args.path is None:
        sessions = [tf.Session(graph=graph) for _ in range(POPULATION_SIZE)]
    

    for sess in sessions:
        sess.run(init)
    

    generation_counter = 0
    while True:
        generation_counter += 1
        fitness_data = []

        file = None

        if generation_counter % RECORD_GEN == 0:
            file = open('games_played/' + str(generation_counter), 'w')
            file.write('generation ' + str(generation_counter) + '\n')
        for i in range(len(sessions)//2):
            ai1 = ai.AIPlayer(sessions[i], Y_index, Y_, W1, B1, W2, B2, W3, B3, W4, B4, X, generation_counter)
            ai2 = ai.AIPlayer(sessions[2*i], Y_index, Y_, W1, B1, W2, B2, W3, B3, W4, B4, X, generation_counter)
            if generation_counter % RECORD_GEN == 0:
                file.write('####### ' + str(i) + ' #######\n')
            player1_wins = play(ai1, ai2, generation_counter, file)

            if generation_counter % RECORD_GEN == 0:
                if player1_wins:
                    file.write('player1 wins\n')
                else:
                    file.write('player2 wins\n')

            if not player1_wins:
                temp = sessions[i]
                sessions[i] = sessions[2*i]
                sessions[i] = temp

        if is_training_finished:
            break

        for sess in sessions[POPULATION_SIZE//2:]:
            sess.close()
        del sessions[POPULATION_SIZE//2:]


        #mutated
        for index in range(0, POPULATION_SIZE//4):
            sess = tf.Session(graph=graph)
            sess.run(init)
            if np.random.random_sample() < MUTATION_PROBABILITY:
                if np.random.random_sample() < W_MUTATION_PROBABILITY:
                    w1_, w2_, w3_, w4_ = sessions[index].run([W1, W2, W3, W4])
                    w1_ = mutate_w_with_percent_change(w1_)
                    w2_ = mutate_w_with_percent_change(w2_)
                    w3_ = mutate_w_with_percent_change(w3_)
                    w4_ = mutate_w_with_percent_change(w4_)
                    sess.run([W1_assign, W2_assign, W3_assign, W4_assign],feed_dict={W1_placeholder:w1_, W2_placeholder:w2_, W3_placeholder:w3_, W4_placeholder:w4_})

                if  np.random.random_sample() < B_MUTATION_PROBABILITY:
                    b1_, b2_, b3_, b4_ = sessions[index].run([B1, B2, B3, B4])
                    b1_ = mutate_b_with_percent_change(b1_)
                    b2_ = mutate_b_with_percent_change(b2_)
                    b3_ = mutate_b_with_percent_change(b3_)
                    b4_ = mutate_b_with_percent_change(b4_)
                    sess.run([B1_assign, B2_assign],feed_dict={B1_placeholder:b1_, B2_placeholder:b2_, B3_placeholder:b3_, B4_placeholder:b4_})

            sessions.append(sess)

        #crossover
        for index in range(0, POPULATION_SIZE//4):
            sess = tf.Session(graph=graph)
            sess.run(init)
            w11_, w12_, w13_, w14_, b11_, b12_, b13_, b14_ = sessions[index].run([W1, W2, W3, W4, B1, B2, B3, B4])
            w21_, w22_, w23_, w24_, b21_, b22_, b23_, b24_ = sessions[index*2].run([W1, W2, W3, W4, B1, B2, B3, B4])

            w1_, w2_, w3_, w4_, b1_, b2_, b3_, b4_ = cross_over(w11=w11_, w12=w12_, w13=w13_, w14=w14_, b11=b11_, b12=b12_, b13=b13_, b14=b14_, w21=w21_, w22=w22_, w23=w23_, w24=w24_, b21=b21_, b22=b22_, b23=b23_, b24=b24_)
            sess.run([W1_assign, W2_assign, W3_assign, W4_assign, B1_assign, B2_assign, B3_assign, B4_assign], 
                    feed_dict={W1_placeholder:w1_, W2_placeholder:w2_, W3_placeholder:w3_, W4_placeholder:w4_, B1_placeholder:b1_, B2_placeholder:b2_, B3_placeholder:b3_, B4_placeholder:b4_})

            sessions.append(sess)

        if generation_counter % RECORD_GEN == 0:
            for i, sess in enumerate(sessions):
                saver.save(sess, 'game_checkpoints/' + str(i))
            
            file.close()




def play(player1, player2, generation, file):
    game = Game(player1, player2)
    game.player1_turn = random.randint(0, 1) == 0

    while not game.game_over():
        if generation % RECORD_GEN == 0:
            game.print_board(file)

        move = -1
        while not game.is_valid_move(move):
            if game.player1_turn == True:
                move, random_move = game.player1.get_move(game)
            else:
                move, random_move = game.player2.get_move(game)

            if not game.is_valid_move(move) and random_move == False:
                game.player1_turn = not game.player1_turn

                if generation % RECORD_GEN == 0:
                    file.write('random == False\n')

                return game.player1_turn

        if generation % RECORD_GEN == 0:
            file.write('random == ' + str(random_move) + '\n')

        game.add_move(move)

        game.player1_turn = not game.player1_turn

    game.player1_turn = not game.player1_turn
    if generation % RECORD_GEN == 0:
        game.print_board(file)

    return game.get_winner



if __name__ == '__main__':
    main()