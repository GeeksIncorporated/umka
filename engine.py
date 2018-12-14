#!/home/alexeyv/workspace/umka/venv/bin/python

import chess
import sys

import datetime
from chess.uci import Engine

from coach.trainer import Coach
from main import UmkaEvaluator
from minimax import MiniMax
from model.model import UmkaNeuralNet, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

MODEL = UmkaNeuralNet(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE)

DEPTH = 3


class UmkaEngine(Engine):

    def move(self, m):
        self.board.push(m)
        print("move %s" % m)

    def on_line_received(self, l):
        # super(UmkaEngine, self).on_line_received(buf)
        if l == 'xboard':
            print('feature myname="Umka" setboard=1 done=1')
        elif l == 'quit':
            sys.exit(0)
        elif l == 'new':
            self.coach = Coach(MODEL)
            self.evaluator = UmkaEvaluator(self.coach.get_model())
            self.brain = MiniMax(self.evaluator)
            self.board = chess.Board()

        elif l == 'uci':
            print("id name Umka")
            print("id author Martin C. Doege")
            print("option name maxplies type spin default 1 min 0 max 1024")
            print("option name qplies type spin default 7 min 0 max 1024")
            print("option name pstab type spin default 0 min 0 max 1024")
            print("option name matetest type check default true")
            print("option name MoveError type spin default 0 min 0 max 1024")
            print(
                "option name BlunderError type spin default 0 min 0 max 1024")
            print(
                "option name BlunderPercent type spin default 0 min 0 max 1024")
            print("uciok")

        elif l.startswith('go') or l == 'force':
            if not self.board:
                self.board = chess.Board()
            m = self.brain.run(self.board, depth=DEPTH)
            self.move(m)

        else:
            if not self.board:
                self.board = chess.Board()
            try:
                self.board.push_uci(l)
                m = self.brain.run(self.board, depth=DEPTH)
                self.move(m)
            except:
                pass


if __name__ == "__main__":
    umka = UmkaEngine()
    sys.stdin.flush()

    while True:
        l = ''
        try:
            l = input()
            umka.on_line_received(l)
        except KeyboardInterrupt:
            try:
                sys.stdin.flush()
            except:
                pass
