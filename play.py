import chess.svg
from chess.pgn import Game
from core.minimax import MiniMax
# from core.minimax_optimized import MiniMaxOpt
from core.umka import Umka
from settings import PATH_TO_MODEL, DEPTH


def play(brain):
    board = chess.Board()

    while not board.is_game_over():
        move = brain.run(board, DEPTH)
        board.push(move)
        print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

    game = Game().from_board(board)
    print(game)


if __name__ == "__main__":
    umka = Umka(path=PATH_TO_MODEL, training_enabled=False)
    brain = MiniMax(umka)
    play(brain)
