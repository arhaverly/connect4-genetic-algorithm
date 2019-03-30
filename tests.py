from connect4 import *

def test_logic():
    game = Game(None, None)
    board = [
                [0, 2, 0, 0, 0, 0, 0],
                [0, 0, 2, 1, 0, 0, 0],
                [0, 0, 1, 2, 0, 0, 0],
                [0, 2, 0, 0, 2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]

    game.set_board(board)
    game.print_board()
    print(game.game_over())


def main():
    test_logic()

if __name__ == '__main__':
    main()