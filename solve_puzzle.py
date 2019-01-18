import pprint
import time
import chess.svg
from chess.pgn import Game
from core.minimax_id import MiniMaxIterativeDeepening
from core.umka import Umka


def solve(fen):
    umka = Umka(path="core/models/model.pth.tar", training_enabled=False)
    brain = MiniMaxIterativeDeepening(umka)

    board = chess.Board(fen)

    while not board.is_game_over():
        move = brain.make_move(board, time_to_think=15*100)
        print(move, brain.best_val, brain.best_move, brain.root_moves)
        board.push(move)
        pprint.pprint(str(board))

    return str(chess.pgn.Game().from_board(board)).split("\n\n")[1]


if __name__ == "__main__":
    with open("data/puzzles/mate_in_2") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 5):
            desc = lines[i]
            fen = lines[i+1]
            res = lines[i+2]
            st = time.time()
            sol = solve(fen)
            print("SOLVES:", sol)
            print("EXPECT:", res)
            print(time.time() - st)
            time.sleep(5)