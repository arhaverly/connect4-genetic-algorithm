import random
from game import *
import human
import ai

import tensorflow as tf
import numpy as np
import argparse

from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

RECORD_GEN = 1

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


def cross_over(weights, biases):
    new_weights = []
    for w1, w2 in weights:
        new_w = []
        len_w1 = len(w1)
        for i in range(len_w1):
            row = []
            len_w1_0 = len(w1[0])
            for j in range(len_w1_0):
                if np.random.random_sample() > 0.5:
                    row.append(w1[i][j])
                else:
                    row.append(w2[i][j])
            new_w.append(row)

        new_weights.append(new_w)
    
    new_biases = []

    for b1, b2 in biases:
        new_b = []
        len_b1 = len(b1)
        for i in range(len_b1):
            if np.random.random_sample() > 0.5:
                new_b.append(b1[i])
            else:
                new_b.append(b2[i])

        new_biases.append(new_b)


    return new_weights, new_biases
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restore', action='store_true')

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

        
    if args.restore is False:
        sessions = [tf.Session(graph=graph) for _ in range(POPULATION_SIZE)]
    else:
        sessions = []
        tf.reset_default_graph() 

        # imported_meta = tf.train.import_meta_graph("game_checkpoints/20.meta")  
        # with tf.Session() as sess:
        # sess = tf.Session()
        # imported_meta = tf.train.import_meta_graph("game_checkpoints/20.meta")  
        # imported_meta.restore(sess, tf.train.latest_checkpoint('game_checkpoints/'))
        # sess.run(init)
        # sessions.append(sess)
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(sess, ['0'], 'game_checkpoints')


    # print(sessions)

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

        new_sessions = []

        #mutated
        for index in range(0, POPULATION_SIZE//2):
            sess = tf.Session(graph=graph)
            sess.run(init)
            if np.random.random_sample() < MUTATION_PROBABILITY:
                if np.random.random_sample() < W_MUTATION_PROBABILITY:
                    w1_, w2_, w3_, w4_ = sessions[index].run([W1, W2, W3, W4])
                    w1_ = mutate_w_with_percent_change(w1_)
                    w2_ = mutate_w_with_percent_change(w2_)
                    w3_ = mutate_w_with_percent_change(w3_)
                    w4_ = mutate_w_with_percent_change(w4_)
                    # sess.run([W1_assign, W2_assign, W3_assign, W4_assign],feed_dict={W1_placeholder:w1_, W2_placeholder:w2_, W3_placeholder:w3_, W4_placeholder:w4_})

                if  np.random.random_sample() < B_MUTATION_PROBABILITY:
                    b1_, b2_, b3_, b4_ = sessions[index].run([B1, B2, B3, B4])
                    b1_ = mutate_b_with_percent_change(b1_)
                    b2_ = mutate_b_with_percent_change(b2_)
                    b3_ = mutate_b_with_percent_change(b3_)
                    b4_ = mutate_b_with_percent_change(b4_)
                    # sess.run([B1_assign, B2_assign],feed_dict={B1_placeholder:b1_, B2_placeholder:b2_, B3_placeholder:b3_, B4_placeholder:b4_})

            new_sessions.append(sess)

        #10
        for index in range(0, POPULATION_SIZE//4):
            sess = tf.Session(graph=graph)
            sess.run(init)
            w11_, w12_, w13_, w14_, b11_, b12_, b13_, b14_ = sessions[index].run([W1, W2, W3, W4, B1, B2, B3, B4])
            w21_, w22_, w23_, w24_, b21_, b22_, b23_, b24_ = sessions[index*2].run([W1, W2, W3, W4, B1, B2, B3, B4])

            weights = [[w11_, w21_], [w12_, w22_], [w13_, w23_], [w14_, w24_]]
            biases = [[b11_, b21_], [b12_, b22_], [b13_, b23_], [b14_, b24_]]
            new_weights, new_biases = cross_over(weights, biases)
            w1_, w2_, w3_, w4_ = new_weights
            b1_, b2_, b3_, b4_ = new_biases
            
            if np.random.random_sample() < W_MUTATION_PROBABILITY:
                w1_, w2_, w3_, w4_ = sessions[index].run([W1, W2, W3, W4])
                w1_ = mutate_w_with_percent_change(w1_)
                w2_ = mutate_w_with_percent_change(w2_)
                w3_ = mutate_w_with_percent_change(w3_)
                w4_ = mutate_w_with_percent_change(w4_)

            if np.random.random_sample() < B_MUTATION_PROBABILITY:
                b1_, b2_, b3_, b4_ = sessions[index].run([B1, B2, B3, B4])
                b1_ = mutate_b_with_percent_change(b1_)
                b2_ = mutate_b_with_percent_change(b2_)
                b3_ = mutate_b_with_percent_change(b3_)
                b4_ = mutate_b_with_percent_change(b4_)

            sess.run([W1_assign, W2_assign, W3_assign, W4_assign, B1_assign, B2_assign, B3_assign, B4_assign], 
                    feed_dict={W1_placeholder:w1_, W2_placeholder:w2_, W3_placeholder:w3_, W4_placeholder:w4_, B1_placeholder:b1_, B2_placeholder:b2_, B3_placeholder:b3_, B4_placeholder:b4_})

            new_sessions.append(sess)

        for i in range(1, 5):
            for index in range(0, POPULATION_SIZE//4-i):
                sess = tf.Session(graph=graph)
                sess.run(init)
                w11_, w12_, w13_, w14_, b11_, b12_, b13_, b14_ = sessions[index].run([W1, W2, W3, W4, B1, B2, B3, B4])
                w21_, w22_, w23_, w24_, b21_, b22_, b23_, b24_ = sessions[index+i].run([W1, W2, W3, W4, B1, B2, B3, B4])

                weights = [[w11_, w21_], [w12_, w22_], [w13_, w23_], [w14_, w24_]]
                biases = [[b11_, b21_], [b12_, b22_], [b13_, b23_], [b14_, b24_]]
                new_weights, new_biases = cross_over(weights, biases)
                w1_, w2_, w3_, w4_ = new_weights
                b1_, b2_, b3_, b4_ = new_biases
                
                if np.random.random_sample() < W_MUTATION_PROBABILITY:
                    w1_, w2_, w3_, w4_ = sessions[index].run([W1, W2, W3, W4])
                    w1_ = mutate_w_with_percent_change(w1_)
                    w2_ = mutate_w_with_percent_change(w2_)
                    w3_ = mutate_w_with_percent_change(w3_)
                    w4_ = mutate_w_with_percent_change(w4_)

                if np.random.random_sample() < B_MUTATION_PROBABILITY:
                    b1_, b2_, b3_, b4_ = sessions[index].run([B1, B2, B3, B4])
                    b1_ = mutate_b_with_percent_change(b1_)
                    b2_ = mutate_b_with_percent_change(b2_)
                    b3_ = mutate_b_with_percent_change(b3_)
                    b4_ = mutate_b_with_percent_change(b4_)

                sess.run([W1_assign, W2_assign, W3_assign, W4_assign, B1_assign, B2_assign, B3_assign, B4_assign], 
                        feed_dict={W1_placeholder:w1_, W2_placeholder:w2_, W3_placeholder:w3_, W4_placeholder:w4_, B1_placeholder:b1_, B2_placeholder:b2_, B3_placeholder:b3_, B4_placeholder:b4_})

                new_sessions.append(sess)


        if generation_counter % RECORD_GEN == 0:
            print("Generation #" + str(generation_counter))
        #     builder = tf.saved_model.builder.SavedModelBuilder('builds')
        #     signature = predict_signature_def(inputs={'myInput': W1},
        #                                       outputs={'myOutput': A1})
        #     # using custom tag instead of: tags=[tag_constants.SERVING]
        #     sess = sessions[0]
        #     builder.add_meta_graph_and_variables(sess=sess,
        #                                          tags=['0'],
        #                                          signature_def_map={'predict': signature})
        #     builder.save()
            
        #     file.close()

        for sess in sessions:
            sess.close()

        sessions = new_sessions


def play(player1, player2, generation, file):
    game = Game(player1, player2)
    game.player1_turn = random.randint(0, 1) == 0

    while not game.game_over():
        if generation % RECORD_GEN == 0:
            game.print_board(file)

        move = -1
        while not game.is_valid_move(move):
            if game.player1_turn == True:
                move = game.player1.get_move(game)
            else:
                move = game.player2.get_move(game)

            if not game.is_valid_move(move):
                game.player1_turn = not game.player1_turn
                return game.player1_turn

        game.add_move(move)

        game.player1_turn = not game.player1_turn

    game.player1_turn = not game.player1_turn
    if generation % RECORD_GEN == 0:
        game.print_board(file)

    return game.player1_turn



if __name__ == '__main__':
    main()