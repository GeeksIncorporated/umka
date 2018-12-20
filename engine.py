#!/home/alexeyv/workspace/umka/venv/bin/python

import os
import signal
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

        elif l == 'white':
            self.board.turn = True

        elif l == 'black':
            self.board.turn = False

        elif l.startswith('time'):
            pass

        elif l.startswith('otim'):
            pass

        elif l.startswith('computer'):
            pass

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

        elif l.startswith('post'):
            pass

        elif l.startswith('hard'):
            pass
        elif l.startswith('random'):
            pass
        elif l.startswith('level'):
            pass

        elif l.startswith('go'):
            m = self.brain.run(self.board, depth=DEPTH)
            self.board.push(m)
            print("move %s" % m)

        else:
            try:
                self.board.push_uci(l)
                m = self.brain.run(self.board, depth=DEPTH)
                self.board.push(m)
                print("move %s" % m)
            except:
                # print("Command:", sys.exc_info())
                pass

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    umka = UmkaEngine()
    sys.stdin.flush()
    l = ''
    while True:
        try:
            l = input()
            umka.on_line_received(l)
            sys.stdin.flush()
        except KeyboardInterrupt:
            print("Input 1: %s %s" % (sys.exc_info(), l))
            try:
                sys.stdin.flush()
                pass
            except:
                print("Input 2:", sys.exc_info())
                pass
