import chess
import torch

from coach.trainer import board_tensor, Coach
from minimax import MiniMax
from model.model import UmkaNeuralNet, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from utils import show_board


class UmkaEvaluator:
    def __init__(self, model):
        self.model = model

    def get_score(self, board):
        sample = board_tensor(board=board)
        input = torch.FloatTensor(sample)
        evaluation = self.model(input)
        score = sum(sample)/10 + evaluation.item()
        #show_board(board)
        #print('Board score:', score)
        return score

if __name__ == "__main__":
    model = UmkaNeuralNet(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE)

    board = chess.Board()

    with Coach(model) as coach:
        evaluator = UmkaEvaluator(coach.get_model())

        while True:
            board.reset()
            while not board.is_game_over():
                move = MiniMax(evaluator).run(board, depth=2)
                board.push(move)

            print(chess.pgn.Game().from_board(board))

            f = open("/tmp/last.pgn", "w")
            f.write(str(chess.pgn.Game().from_board(board)))
            f.close()
            coach.train_pgn("/tmp/last.pgn")
            board.mirror()
