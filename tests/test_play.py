import unittest

import chess
from chess.pgn import Game

from core.minimax import MiniMax
from core.umka import Umka
from settings import PATH_TO_MODEL, DEPTH


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.umka = Umka(path=PATH_TO_MODEL, training_enabled=False)

    def test_minimax(self):
        board = chess.Board()
        brain = MiniMax(self.umka)
        while not board.is_game_over():
            move = brain.run(board, DEPTH)
            board.push(move)
            print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

        game = Game().from_board(board)
        print(game)

