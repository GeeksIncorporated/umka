import sys
import time
from chess.polyglot import zobrist_hash


class MiniMax:
    def __init__(self, umka):
        self.umka = umka
        self.main_line = {}
        self.last_time_info_printed = 0

    def min_play(self, board, depth, alpha, beta):
        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if board.is_game_over() or depth <= 0:
            score = self.umka.evaluate(board)
            self.cache[zobrist_hash(board)] = score
            return score

        value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = min(value, self.max_play(board, depth - 1, alpha, beta))
            board.pop()
            if value < alpha:
                self.print_info(depth, move, value)
                return value
            beta = min(beta, value)
        return value

    def max_play(self, board, depth, alpha, beta):
        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if board.is_game_over() or depth <= 0:
            score = self.umka.evaluate(board)
            self.cache[zobrist_hash(board)] = score
            return score

        value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            value = max(value, self.min_play(board, depth - 1, alpha, beta))
            board.pop()
            if value > beta:
                self.print_info(depth, move, value)
                return value
            alpha = max(alpha, value)
        return value

    def run(self, board, depth):
        self.nodes = 0
        self.cache = {}
        self.st = time.time()
        best_val = float('-inf')
        beta = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            self.print_info(depth, move, 0)
            value = self.min_play(board, depth - 1, best_val, beta)
            board.pop()
            if value > best_val:
                best_val = value
                best_move = move
                self.print_info(depth, move, best_val)
        return best_move

    def print_info(self, depth, move, cp):
        if (time.time() - self.last_time_info_printed) < 0.5:
            return
        self.main_line[depth] = move
        main_line = ' '.join(
            [str(v) for k, v in reversed(sorted(self.main_line.items()))][:depth+1])
        move_time = time.time() - self.st
        t = int(move_time) * 1000

        print("info time %12s nodes %12s nps %12s cp %12s pv %s" % (
            t, self.nodes, int(self.nodes / move_time), int(1000 * cp), main_line))
        self.last_time_info_printed = time.time()
        sys.stdout.flush()
