import sys
import time
from multiprocessing.dummy import Pool

from chess.polyglot import zobrist_hash
from core.utils import position_needs_attention
from settings import DEPTH, CONCURRENTS, MAX_DEPTH


class MiniMaxParallel:
    def __init__(self, umka):
        self.umka = umka
        self.best_move = "N/A"
        self.best_val = -1000
        self.last_time_info_printed = 0

    def play(self, board, depth, alpha, beta, maximize, need_attention, d):
        self.nodes += 1

        # if zobrist_hash(board) in self.cache:
        #     return self.cache[zobrist_hash(board)]

        if need_attention and d < MAX_DEPTH:
            depth -= 1

        if board.is_game_over() or depth >= DEPTH:
            score = self.umka.evaluate(board)
            self.print_info(d)
            # self.cache[zobrist_hash(board)] = score
            return score

        moves = board.legal_moves

        if maximize:
            alpha = float('-inf')
            for move in moves:
                # need_attention = False
                need_attention = position_needs_attention(board, move)
                board.push(move)
                alpha = max(alpha,
                            self.play(
                                board, depth + 1, alpha, beta,
                                False, need_attention, d + 1))
                board.pop()
                if beta <= alpha:
                    return alpha
            return alpha
        else:
            beta = float('inf')
            for move in moves:
                # need_attention = False
                need_attention = position_needs_attention(board, move)
                board.push(move)
                beta = min(beta,
                            self.play(board, depth + 1, alpha, beta,
                                      True, need_attention, d + 1))
                board.pop()
                if beta <= alpha:
                    return beta
            return beta

    def run(self, board, depth=0):

        self.nodes = 0
        self.st = time.time()
        self.max_depth = depth
        self.cache = {}

        if board.turn:
            beta = float('inf')
            self.best_val = -1000
            jobs = []

            def _proccess_job(job):
                d = 0
                board, move, best_val, beta = job
                # need_attention = False
                need_attention = position_needs_attention(board, move)
                board.push(move)

                value = self.play(
                    board, depth + 1, best_val, beta, True, need_attention, d + 1)
                board.pop()
                if value > self.best_val:
                    self.best_val = value
                    self.best_move = move
                return (self.best_val, time.time(), self.best_move)
            for move in board.legal_moves:
                jobs.append((board.copy(), move, self.best_val, beta))
            pool = Pool(CONCURRENTS)
            moves = pool.map(_proccess_job, jobs)
            return max(moves)[2]
        else:
            jobs = []
            beta = float('inf')
            self.best_val = 1000

            def _proccess_job_min(job):
                d = 0
                board, move, best_val, beta = job
                # need_attention = False
                need_attention = position_needs_attention(board, move)
                board.push(move)
                value = self.play(
                    board, depth + 1, best_val, beta, False, need_attention, d + 1)
                board.pop()

                if value < self.best_val:
                    self.best_val = value
                    self.best_move = move
                return (self.best_val, time.time(), self.best_move)

            for move in board.legal_moves:
                jobs.append((board.copy(), move, self.best_val, beta))
            pool = Pool(CONCURRENTS)
            moves = pool.map(_proccess_job_min, jobs)
            return min(moves)[2]

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
