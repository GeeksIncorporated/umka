from IPython.core.display import SVG
import chess.svg
from chess.pgn import Game
from core.minimax import MiniMax
from core.umka import Umka

umka = Umka(path="model/model.pth.tar", training_enabled=False)
brain = MiniMax(umka)
board = chess.Board()


while not board.is_game_over():
    move = brain.run(board, depth=3)
    board.push(move)
    SVG(chess.svg.board(board=board, size=400))
    print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

game = Game().from_board(board)
print(game)
