import chess
from chess.pgn import Game
from core.minimax import MiniMax
from core.umka import Umka

umka = Umka(path="model/model.pth.tar", training_enabled=False)
brain = MiniMax(umka)
board = chess.Board()


while not board.is_game_over():
    move = brain.run(board, depth=2)
    board.push(move)

game = Game().from_board(board)
print(game)
