#!/home/alexeyv/workspace/umka/venv/bin/python

import os
import signal
import sys

import chess
from chess.uci import Engine

from core.minimax_id import MiniMaxIterativeDeepening
from core.umka import Umka
from settings import ENGINE_TIME


class UmkaEngine(Engine):

    def __init__(self):
        self.board = None
        self.time_to_think = 1200 * 100
        self.path = os.path.join("core/models/model.pth.tar")

    def on_line_received(self, l):
        # super(UmkaEngine, self).on_line_received(buf)

        if l == 'xboard':
            print('feature myname="Umka" setboard=1 done=1 sigint=0 sigterm=0')
            print('done')

        elif l == 'quit':
            sys.exit(0)

        elif l == 'new':
            self.board = chess.Board()
            self.umka = Umka(
                self.path, training_enabled=False)
            self.brain = MiniMaxIterativeDeepening(self.umka)

        elif l == 'white':
            m = self.make_move()
            self.board.push(m)
            print("move %s" % m)

        elif l == 'black':
            pass

        elif l.startswith('time'):
            self.time_to_think = 120 * 100 #float(l.split(" ")[1]) / 10
            print(self.time_to_think)

        elif l.startswith('otim'):
            pass
        elif l.startswith('force'):
            pass
        elif l.startswith('computer'):
            pass
        elif l.startswith('undo'):
            self.board.pop()
        elif l.startswith('setboard'):
            self.board = chess.Board(fen=" ".join(l.split(" ")[1:]))
        elif l.startswith('result'):
            pass

        elif l == 'uci':
            print("id name Umka")
            print("id author GeeksIncorporated")
            print("option name maxplies type spin default 1 min 0 max 1024")
            print("option name qplies type spin default 7 min 0 max 1024")
            print("option name pstab type spin default 0 min 0 max 1024")
            print("option name matetest type check default true")
            print("option name MoveError type spin default 0 min 0 max 1024")
            print(
                "option name BlunderError type  spin default 0 min 0 max 1024")
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
            pass

        else:
            try:
                self.board.push_uci(l)
                m = self.make_move()
                self.board.push(m)
                print("move %s" % m)
            except:
                print("Command:", sys.exc_info())
                print(self.board)
                pass

    def make_move(self):
        m = self.brain.umka.get_move_from_opennings(self.board)
        if not m:
            m = self.brain.make_move(self.board, time_to_think=self.time_to_think)
        return m


def signal_handler(*args, **kwargs):
    print('You pressed Ctrl+C!')

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIG_IGN, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
        sys.stdout.flush()