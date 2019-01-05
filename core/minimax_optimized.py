import sys
import time
from chess.polyglot import zobrist_hash


class MiniMaxOpt:
    def __init__(self, umka):
        self.umka = umka
        self.main_line = {}
        self.last_time_info_printed = 0
        self.nodes = 0
        self.best_move = None
        self.cache = {}

    def alpha_beta_with_memory(self, board, depth, alpha, beta, maximize):
        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if board.is_game_over() or depth <= 0:
            val = self.umka.evaluate(board)
            self.cache[zobrist_hash(board)] = val
            return val

        if maximize:
            value = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                value = max(value, self.alpha_beta_with_memory(
                    board, depth - 1, alpha, beta, False))
                board.pop()
                if value > beta:
                    self.main_line[depth] = move
                    self.print_info(depth, move, value)
                    return value
                alpha = max(alpha, value)
            return value
        else:
            value = float('inf')
            for move in board.legal_moves:
                board.push(move)
                value = min(value, self.alpha_beta_with_memory(
                    board, depth - 1, alpha, beta, True))
                board.pop()
                if value < alpha:
                    self.main_line[depth] = move
                    # self.print_info(depth, move, value)
                    return value
                beta = min(beta, value)
            return value

    def MDTf(self, board, f, depth, maximize):
        g = f
        self.cache = {}
        upperBound = float('inf')
        lowerBound = float('-inf')
        while lowerBound < upperBound:
            b = max(g, lowerBound + 0.01)
            g = self.alpha_beta_with_memory(board, depth, b - 0.1, b, maximize)
            if g < b:
                upperBound = g
            else:
                lowerBound = g
        return g

    def run(self, board, maximize):
        self.st = time.time()
        firstguess = 0
        for d in range(100):
            firstguess = self.MDTf(board, firstguess, d, maximize)
        return self.main_line[1]

    def print_info(self, depth, move, cp):
        if (time.time() - self.last_time_info_printed) < 0.5:
            return
        self.main_line[depth] = move
        main_line = ' '.join(
            [str(v) for k, v in reversed(sorted(self.main_line.items()))][
            :depth + 1])
        move_time = time.time() - self.st
        t = int(move_time) * 1000

        print("info time %12s nodes %12s nps %12s cp %12s pv %s" % (
            t, self.nodes, int(self.nodes / move_time), int(1000 * cp),
            main_line))
        self.last_time_info_printed = time.time()
        sys.stdout.flush()
