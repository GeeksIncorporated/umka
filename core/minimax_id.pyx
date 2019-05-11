# cython: language_level=3

import bisect
import gc
import itertools
import sys
import time
from copy import deepcopy

from chess.pgn import Game

from chess.polyglot import zobrist_hash

from chess import Board, Move, bbin
from settings import DEPTH, CHECKMATE

INF = 10000

# TODO : cythonize all this shit
cdef class MiniMaxIterativeDeepening:
    cdef public object umka
    cdef public object best_move
    cdef public float best_val
    cdef long last_time_info_printed
    cdef long st
    cdef int time_to_think
    cdef int max_depth
    cdef int nodes
    cdef object cache
    cdef public object root_moves

    def __init__(self, umka):
        self.umka = umka
        self.best_move = None
        self.best_val = 0
        self.last_time_info_printed = 0
        self.time_to_think = 60 * 100  # sec

    def time_is_up(self):
        # return
        return (time.time() - self.st) > self.time_to_think

    def _minimax(self, board, depth, alpha, beta, maximize):
        zh = zobrist_hash(board)
        if zh in self.cache:
            return self.cache[zh]

        if depth >= self.max_depth - 1:
            boards = []
            for move in board.legal_moves:
                self.nodes += 1
                board.push(move)
                boards.append(deepcopy(board))
                board.pop()
            if len(boards) == 0:
                if board.is_stalemate():
                    return 0
                elif board.is_checkmate():
                    return CHECKMATE
            elif board.can_claim_draw():
                return 0

            score = self.umka.evaluate_bulk(boards, depth, not maximize)
            self.print_info(depth + 1, board)
            self.cache[zobrist_hash(board)] = score
            return score

        if maximize:
            value = INF
            for move in board.legal_moves:
                # if self.time_is_up():
                #     return -INF
                board.push(move)
                value = min(value,
                            self._minimax(
                                board, depth + 1, alpha, beta, False))
                board.pop()

                if value <= alpha:
                    break

                beta = min(beta, value)
        else:
            value = -INF
            for move in board.legal_moves:
                board.push(move)
                value = max(value,
                            self._minimax(
                                board, depth + 1, alpha, beta, True))
                board.pop()

                if value >= beta:
                    break

                alpha = max(alpha, value)
        return value

    def alphabeta_minimax(self, board, depth=0):
        self.nodes = 0
        self.cache = {}
        gc.disable()
        beta = INF
        if board.turn:
            best_val = -INF
            for rm in list(self.root_moves):
                self.root_moves.remove(rm)
                board.push(rm.move)
                # if board.can_claim_draw():
                #     value = 0
                # else:
                value = self._minimax(
                    board, depth + 1, best_val, beta, True)
                board.pop()
                rm.value = -value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if abs(self.best_val) >= CHECKMATE:
                    break
                # self.print_info(depth, board)
        else:
            best_val = -INF
            for rm in list(self.root_moves):
                self.root_moves.remove(rm)
                board.push(rm.move)
                # if board.can_claim_draw():
                #     value = 0
                # else:
                value = self._minimax(
                    board, depth + 1, best_val, beta, False)
                board.pop()
                rm.value = value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if abs(self.best_val) >= CHECKMATE:
                    break
                # self.print_info(depth, board)
        gc.enable()
        gc.collect()

    def print_info(self, depth, board):
        if (time.time() - self.last_time_info_printed) < 2:
            return
        move_time = time.time() - self.st
        t = int(move_time) * 1000
        print(board)
        print(
            "info time= %8s depth= %8s nodes= %8s nps= %8s cp= %6s pv= %s %s %s" % (
                t, depth, self.nodes, int(self.nodes / move_time),
                int(1000 * self.best_val),
                self.best_move, self.root_moves,
                str(Game().from_board(board)).split("\n\n")[1]))
        self.last_time_info_printed = time.time()
        sys.stdout.flush()

    def make_move(self, board, time_to_think=30 * 100):

        self.st = time.time()
        self.time_to_think = time_to_think

        self.root_moves = [
            SortableMove(m) for m in itertools.chain(
            board.legal_moves)]

        d = 1
        self.st = time.time()

        while d <= DEPTH and not self.time_is_up():
            self.max_depth = d
            self.alphabeta_minimax(board)
            d += 1
            print("---------------------->", self.best_move)
            if abs(self.best_val) >= CHECKMATE:
                break

        return self.best_move


class SortableMove(Move):

    def __init__(self, move, float value=INF):
        self.move = move
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return "%s %s" % (self.move, self.value)