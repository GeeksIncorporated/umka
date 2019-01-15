import time
import unittest

import chess
from chess.pgn import Game

from core.minimax import MiniMax
from core.__umka import Umka
from settings import PATH_TO_MODEL, DEPTH
import PyMoveGen as board_c


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

    def test_board_perfomance(self):
        board = chess.Board()
        brain = MiniMax(self.umka)
        while not board.is_game_over():
            move = brain.run(board)
            board.push(move)
            # print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])

        san_moves = []
        for move in board.move_stack:
            san_moves.append(str(move))

        st = time.time()
        board1 = chess.Board()
        res = []
        for m in san_moves:
            board1.push_uci(m)
            res += map(str, board1.legal_moves)
        print(len(res), time.time() - st)



        st = time.time()
        res = []
        board_c.set_startpos()
        for m in san_moves:
            board_c.move(m)
            res += board_c.get_all_moves()
        print(len(res), time.time() - st)
        print(brain.cached)