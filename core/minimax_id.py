import bisect
import itertools
import sys
import time
from copy import copy

from chess import Move
from chess.polyglot import zobrist_hash
from settings import DEPTH

INF = 10000


class MiniMaxIterativeDeepening:
    def __init__(self, umka):
        self.umka = umka
        self.best_move = None
        self.best_val = 0
        self.last_time_info_printed = 0
        self.cached = 0
        self.time_to_think = 60  # sec

    def time_is_up(self):
        return time.time() - self.st > self.time_to_think

    def _minimax(self, board, depth, alpha, beta, maximize):

        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if depth > self.max_depth:
            score = self.umka.evaluate(board)
            self.print_info(depth)
            self.cache[zobrist_hash(board)] = score
            return score

        moves = itertools.chain(
            board.generate_castling_moves(),
            board.generate_legal_captures(),
            board.legal_moves)

        if maximize:
            value = INF
            for move in moves:
                if self.time_is_up():
                    return value
                board.push(move)
                value = min(value,
                            self._minimax(
                                board, depth + 1, alpha, beta, False))
                board.pop()
                if value < alpha:
                    return value

                beta = min(beta, value)
            return value
        else:
            value = -INF
            for move in moves:
                if self.time_is_up():
                    return value
                board.push(move)
                value = max(value,
                            self._minimax(board, depth + 1, alpha, beta, True))
                board.pop()
                if value > beta:
                    return value

                alpha = max(alpha, value)
            return value

    def alphabeta_minimax(self, board, depth=0):

        self.st = time.time()
        self.nodes = 0
        self.cache = {}

        beta = INF
        if board.turn:
            best_val = -INF
            for rm in list(self.root_moves):
                move = rm.move
                self.root_moves.remove(rm)
                board.push(move)
                value = self._minimax(
                    board, depth + 1, best_val, beta, True)
                board.pop()
                if value > best_val:
                    best_val = value
                    self.best_move = move
                    self.best_val = value
                bisect.insort_left(self.root_moves, SortableMove(move, value))
        else:
            best_val = INF
            for rm in list(self.root_moves):
                move = rm.move
                self.root_moves.remove(rm)
                board.push(move)
                value = self._minimax(
                    board, depth + 1, best_val, beta, False)
                board.pop()
                if value < best_val:
                    best_val = value
                    self.best_val = value
                    self.best_move = move

                bisect.insort_left(self.root_moves, SortableMove(move, -value))

        return self.root_moves[0].move

    def print_info(self, depth):
        if (time.time() - self.last_time_info_printed) < 0.5:
            return
        move_time = time.time() - self.st
        t = int(move_time) * 1000

        print(
            "info time= %8s depth= %8s nodes= %8s nps= %8s cp= %6s pv= %s" % (
                t, depth, self.nodes, int(self.nodes / move_time),
                int(1000 * self.best_val),
                self.best_move))
        self.last_time_info_printed = time.time()
        sys.stdout.flush()

    def make_move(self, board, time_to_think=30):

        self.st = time.time()
        self.time_to_think = time_to_think

        self.root_moves = [
            SortableMove(m) for m in itertools.chain(
            board.generate_castling_moves(),
            board.generate_legal_captures(),
            board.legal_moves)]

        d = 3
        while d < DEPTH:
            self.max_depth = d
            self.alphabeta_minimax(board)
            d += 1

        return self.best_move


class SortableMove(Move):
    def __init__(self, move, value=0):
        self.move = move
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return "%s %s" % (self.move, self.value)