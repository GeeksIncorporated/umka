import chess
from coach.trainer import Coach
from engine import MODEL
from main import UmkaEvaluator
from minimax import MiniMax
from utils import show_board

coach = Coach(MODEL)
evaluator = UmkaEvaluator(coach.get_model())
brain = MiniMax(evaluator)
board = chess.Board()


while not board.is_game_over():
    show_board(board)
    move = brain.run(board, depth=2)
    board.push(move)
