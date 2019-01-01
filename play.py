import chess.svg
from chess.pgn import Game

from core.minimax_id import MiniMaxIterativeDeepening
from core.umka import Umka


def play(brain):
    board = chess.Board()

    while not board.is_game_over():
        move = brain.make_move(board, time_to_think=120)
        board.push(move)
        print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

    game = Game().from_board(board)
    print(game)


if __name__ == "__main__":
    umka = Umka(path="core/models/model.pth.tar", training_enabled=False)
    brain = MiniMaxIterativeDeepening(umka)
    play(brain)
