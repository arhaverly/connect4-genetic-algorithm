import argparse
import datetime
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import Sequential
import numpy
import os
import random
import tensorflow as tf

import ai
from game import *
import human

numpy.random.seed(137)

RECORD_GEN = 200
POPULATION_SIZE = 40
HALF_POP = POPULATION_SIZE // 2
QUARTER_POP = POPULATION_SIZE // 4
MUTATION_PROBABILITY = 0.8
W_MUTATION_PROBABILITY = 0.5
B_MUTATION_PROBABILITY = 0.5
MAX_GEN = 10000
N_EPISODE = 10

models = []

def init_models(path):
    if path is None:
        for i in range(POPULATION_SIZE):
            model = Sequential()
            model.add(Dense(100, input_dim = 42, activation='relu'))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(7, activation='sigmoid'))
            models.append(model)
    else:
        for i in range(POPULATION_SIZE):
            json_file = open(path + "model.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(path + str(i) + ".h5")
            models.append(loaded_model)


def mutate(new_weights, model_num):
    weights = numpy.copy(models[model_num].get_weights())

    for layer_num, layer in enumerate(weights):
        if layer_num % 2 == 0:
            rows = len(layer)
            cols = len(layer[0])
            for row in range(rows):
                for col in range(cols):
                    if numpy.random.random_sample() > 0.1:
                        weights[layer_num][row][col] = numpy.random.uniform(-1, 1)

        else:
            cols = len(layer)
            for col in range(cols):
                if numpy.random.random_sample() > 0.1:
                    weights[layer_num][col] = numpy.random.uniform(-1, 1)

    new_weights.append(weights)


def crossover(new_weights, model_num1, model_num2):
    weights1 = numpy.copy(models[model_num1].get_weights())
    weights2 = models[model_num2].get_weights()

    for layer_num, layer in enumerate(weights1):
        if layer_num % 2 == 0:
            rows = len(layer)
            cols = len(layer[0])
            for row in range(rows):
                for col in range(cols):
                    if numpy.random.random_sample() > 0.5:
                        weights1[layer_num][row][col] = weights2[layer_num][row][col]

        else:
            cols = len(layer)
            for col in range(cols):
                if numpy.random.random_sample() > 0.5:
                    weights1[layer_num][col] = weights2[layer_num][col]

    new_weights.append(weights1)


def set_model_weights(new_weights):
    for i, model in enumerate(models):
        model.set_weights(new_weights[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    args = parser.parse_args()

    init_models(args.path)
    generation = 0
    file = None
    while True:
        generation += 1
        if generation % RECORD_GEN == 0:
            file = open('games_played/' + str(generation), 'w')
            file.write('generation ' + str(generation) + '\n')

        #play games
        for i in range(HALF_POP):
            if generation % RECORD_GEN == 0:
                file.write('####### ' + str(i) + ' #######\n')

            player1_wins = play(i, i + HALF_POP, file, generation)

            if generation % RECORD_GEN == 0:
                if player1_wins:
                    file.write('player1 wins\n')
                else:
                    file.write('player2 wins\n')

            #disregard losing model
            if not player1_wins:
                models[i] = models[i + HALF_POP]

        #mutate and/or crossover
        new_weights = []

        for model_num in range(HALF_POP):
            weights_and_biases = models[i].get_weights()
            weights = weights_and_biases[0]
            biases = weights_and_biases[1]

            mutate(new_weights, model_num)

        for model_num in range(QUARTER_POP):
            crossover(new_weights, model_num, model_num * 2)

        randints = numpy.random.randint(HALF_POP, size=(QUARTER_POP, 2))
        for model_num1, model_num2 in randints:
            if model_num1 == model_num2:
                new_weights.append(models[model_num1].get_weights())
            else:
                crossover(new_weights, model_num1, model_num2)

        set_model_weights(new_weights)


        #save nets
        if generation % RECORD_GEN == 0:
            directory = "saved_nets/generation_" + str(generation) + "_" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S-%f") + "/"
            os.mkdir(directory)
            
            model_json = models[0].to_json()
            with open(directory + "model.json", "w") as json_file:
                json_file.write(model_json)

            for i in range(POPULATION_SIZE):
                models[i].save_weights(directory + str(i) + ".h5")

        print("Generation #" + str(generation))


def index_of_max(arr):
    max_val = -1
    max_index = 0
    for inner in arr:
        for i, val in enumerate(inner):
            if val > max_val:
                max_val = val
                max_index = i

    return max_index


def play(index1, index2, file, generation):
    game = Game()
    game.player1_turn = (random.randint(0, 1) == 0)

    while not game.game_over():
        if generation % RECORD_GEN == 0:
            game.print_board(file)

        move = -1
        while not game.is_valid_move(move):
            vector = game.board.reshape(1, 42)
            if game.player1_turn == True:
                hypo = models[index1].predict(vector)
            else:
                hypo = models[index2].predict(vector)

            move = index_of_max(hypo)
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