import time
import pprint
import chess.svg
from chess.pgn import Game
# from core.minimax_id import MiniMaxIterativeDeepening
from core.minimax_id_rand_walk import MiniMaxIterativeDeepening
from core.umka import Umka


def play(brain):
    board = chess.Board()

    while not board.is_game_over():
        print('NEW MOVE:')
        st_move = time.time()
        move = brain.make_move(board, time_to_think=15*100)
        print(move, brain.best_val, brain.best_move, brain.root_moves)
        board.push(move)
        pprint.pprint(str(board))
        print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])
        print('Time on move: {}'.format(time.time()-st_move))
        print('================\n')
    game = Game().from_board(board)
    print(game)

def change_val(val1, val2):
    val1 = 4
    val2 = 12


if __name__ == "__main__":
    # l = [5,17]
    # print("BEFORE: {}".format(l))
    # change_val(l[0], l[1])
    # print("AFTER: {}".format(l))
    import bisect
    import random
    for let in 'abcdefig':
        bisect.insort_right('a', random.randint())
    print(bisect.insort_right())
    # umka = Umka(path="core/models/model.pth.tar", training_enabled=False)
    # brain = MiniMaxIterativeDeepening(umka)
    # st = time.time()
    # play(brain)
    # print(time.time() - st)