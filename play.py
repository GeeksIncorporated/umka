import chess
from chess.pgn import Game
from core.minimax import MiniMax
from core.umka import Umka

umka = Umka(path="model/model.pth.tar", training_enabled=False)
brain = MiniMax(umka)
board = chess.Board()


while not board.is_game_over():
    move = brain.run(board, depth=3)
    board.push(move)
    print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

game = Game().from_board(board)
print(game)
