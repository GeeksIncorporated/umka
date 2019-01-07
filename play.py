import pprint
import time

import chess.svg
from chess.pgn import Game

from core.minimax_id import MiniMaxIterativeDeepening
from core.umka import Umka


def play(brain):
    # board = chess.Board("2bqkbn1/2pppp2/np2N3/r3P1p1/p2N2B1/5Q2/PPPPKPP1/RNB2r2 w KQkq - 0 1")
    board = chess.Board()

    while not board.is_game_over():
        move = brain.make_move(board, time_to_think=15*100)
        print(move, brain.best_val, brain.best_move, brain.root_moves)
        board.push(move)
        pprint.pprint(str(board))
        print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

    game = Game().from_board(board)
    print(game)


if __name__ == "__main__":
    umka = Umka(path="core/models/model.pth.tar", training_enabled=False)
    brain = MiniMaxIterativeDeepening(umka)
    st = time.time()
    play(brain)
    print(time.time() - st)