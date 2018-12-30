import chess.svg
from chess.pgn import Game

from core.minimax import MiniMax
from core.minimax_parallel import MiniMaxParallel
from core.umka import Umka
from settings import PATH_TO_MODEL


def play(brain):
    board = chess.Board()

    while not board.is_game_over():
        move = brain.run(board)
        board.push(move)
        print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

    game = Game().from_board(board)
    print(game)


if __name__ == "__main__":
    umka = Umka(path="core/models/model.pth.tar", training_enabled=False)
    brain = MiniMax(umka)
    play(brain)
