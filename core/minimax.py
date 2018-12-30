import itertools
import sys
import time
from chess.polyglot import zobrist_hash
from core.utils import position_needs_attention
from settings import DEPTH, MAX_DEPTH


class MiniMax:
    def __init__(self, umka):
        self.umka = umka
        self.best_move = "N/A"
        self.best_val = 0
        self.last_time_info_printed = 0

    def play(self, board, depth, alpha, beta, maximize, need_attention, d):
        self.nodes += 1
        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if need_attention:
            depth -= 1

        if d >= MAX_DEPTH:
            depth = MAX_DEPTH

        if depth >= DEPTH:
            score = self.umka.evaluate(board, need_attention)
            self.print_info(d)
            self.cache[zobrist_hash(board)] = score
            return score

        moves = itertools.chain(
                    board.generate_castling_moves(),
                    board.legal_moves)

        if maximize:
            value = float('inf')
            for move in moves:
                need_attention = position_needs_attention(board, move)
                board.push(move)
                value = min(value,
                            self.play(
                                board, depth + 1, alpha, beta,
                                False, need_attention, d + 1))
                board.pop()
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value
        else:
            value = float('-inf')
            for move in moves:
                need_attention = position_needs_attention(board, move)
                board.push(move)
                value = max(value,
                            self.play(board, depth + 1, alpha, beta,
                                      True, need_attention, d + 1))
                board.pop()
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

    def run(self, board, depth=0):
        d = 0
        self.nodes = 0
        self.st = time.time()
        self.cache = {}
        beta = float('inf')
        moves = itertools.chain(
                    board.generate_castling_moves(),
                    board.legal_moves)
        best_move = list(board.legal_moves)[0]
        if board.turn:
            best_val = float('-inf')
            for move in moves:
                need_attention = position_needs_attention(board, move)
                board.push(move)
                value = self.play(
                    board, depth + 1, best_val, beta, True, need_attention, d + 1)
                board.pop()
                if value > best_val:
                    best_val = value
                    best_move = move
                    self.best_move = best_move
                    self.best_val = best_val
        else:
            best_val = float('inf')
            for move in moves:
                need_attention = position_needs_attention(board, move)
                board.push(move)
                value = self.play(
                    board, depth + 1, best_val, beta, False, need_attention, d + 1)
                if value < best_val:
                    best_val = value
                    best_move = move
                    self.best_move = best_move
                    self.best_val = best_val
                board.pop()
        return best_move

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
