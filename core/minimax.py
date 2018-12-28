import sys
import time
from chess.polyglot import zobrist_hash


class MiniMax:
    def __init__(self, umka):
        self.umka = umka
        self.main_line = {}
        self.last_time_info_printed = 0

    def play(self, board, depth, alpha, beta, maximize, is_capture, is_castling):
        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if board.is_game_over() or depth <= 0:
            score = self.umka.evaluate(board, is_capture, is_castling, not bool(board.halfmove_clock%2))
            self.cache[zobrist_hash(board)] = score
            return score

        moves = board.legal_moves

        if maximize:
            value = float('inf')
            for move in moves:
                is_capture = board.is_capture(move)
                is_castling = board.is_castling(move)
                board.push(move)
                value = min(value,
                            self.play(
                                board, depth - 1, alpha, beta,
                                False, is_capture, is_castling))
                board.pop()
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value
        else:
            value = float('-inf')
            for move in moves:
                is_capture = board.is_capture(move)
                is_castling = board.is_castling(move)
                board.push(move)
                value = max(value,
                            self.play(board, depth - 1, alpha, beta,
                                      True, is_capture, is_castling))
                board.pop()
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

    def run(self, board, depth):
        self.nodes = 0
        self.st = time.time()
        self.cache = {}
        beta = float('inf')
        best_move = None
        if board.turn:
            best_val = float('-inf')
            for move in board.legal_moves:
                is_capture = board.is_capture(move)
                is_castling = board.is_castling(move)

                board.push(move)
                value = self.play(
                    board, depth - 1, best_val, beta, True, is_capture, is_castling)
                board.pop()
                if value > best_val:
                    best_val = value
                    best_move = move
                    self.print_info(board.halfmove_clock, best_move, best_val)
        else:
            best_val = float('inf')
            for move in board.legal_moves:
                is_capture = board.is_capture(move)
                is_castling = board.is_castling(move)

                board.push(move)
                value = self.play(
                    board, depth - 1, best_val, beta, False, is_capture, is_castling)
                if value < best_val:
                    best_val = value
                    best_move = move
                    self.print_info(board.halfmove_clock, best_move, best_val)
                board.pop()
        return best_move

    def print_info(self, depth, move, cp):
        if (time.time() - self.last_time_info_printed) < 0.5:
            return
        self.main_line[depth] = move
        main_line = ' '.join(
            reversed([str(v) for k, v in reversed(sorted(self.main_line.items()))]))
        move_time = time.time() - self.st
        t = int(move_time) * 1000

        print("info time %8s nodes %8s nps %8s cp %6s pv %s" % (
            t, self.nodes, int(self.nodes / move_time), int(1000 * cp),
            main_line))
        self.last_time_info_printed = time.time()
        sys.stdout.flush()
