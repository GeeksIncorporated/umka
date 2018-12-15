#!/home/alexeyv/workspace/umka/venv/bin/python

import os
import sys

import chess
from chess.uci import Engine

from settings import MODELS_DIR, DEPTH
from core.minimax import MiniMax
from core.umka import Umka


class UmkaEngine(Engine):

    def __init__(self):
        self.board = None
        self.path = os.path.join(MODELS_DIR, "model.pth.tar")

    def move(self, m):
        self.board.push(m)
        print("move %s" % m)

    def on_line_received(self, l):
        # super(UmkaEngine, self).on_line_received(buf)

        if l == 'xboard':
            print('feature myname="Umka" setboard=1 done=1 sigint=0 sigterm=0')

        elif l == 'quit':
            sys.exit(0)

        elif l == 'new':
            self.board = chess.Board()

            self.umka = Umka(
                self.path, training_enabled=False)

            self.brain = MiniMax(self.umka)

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

        elif l.startswith('go'):
            m = self.brain.run(self.board, depth=DEPTH)
            self.move(m)

        else:
            try:
                self.board.push_uci(l)
                m = self.brain.run(self.board, depth=DEPTH)
                self.move(m)
            except:
                print("Command:", sys.exc_info())


if __name__ == "__main__":
    umka = UmkaEngine()
    sys.stdin.flush()

    while True:
        l = ''
        print(1)
        try:
            l = input()
            umka.on_line_received(l)
        except KeyboardInterrupt:
            print("Input 1: %s %s" % (sys.exc_info(),l))
            try:
                sys.stdin.flush()
            except:
                print("Input 2:", sys.exc_info())
