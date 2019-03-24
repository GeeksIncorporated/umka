import bisect
import itertools
import sys
import time
import random
from copy import deepcopy

import chess
from chess import Move
from chess.polyglot import zobrist_hash
from settings import DEPTH, CHECKMATE

INF = 10000


class MiniMaxIterativeDeepening:
    def __init__(self, umka):
        self.umka = umka
        self.best_move = None
        self.best_val = 0
        self.last_time_info_printed = 0
        self.cached = 0
        self.time_to_think = 60 * 100  # sec

    def time_is_up(self):
        # return
        return (time.time() - self.st) > self.time_to_think

    def _random_walk_minimax(self, board, depth, alpha, beta, maximize):
        # print('depth: {}'.format(depth ))
        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)], alpha, beta

        if depth >= self.max_depth or len(list(board.legal_moves)) == 0:
            score = self.umka.evaluate(board, depth, maximize)
            # self.print_info(depth, board)
            self.cache[zobrist_hash(board)] = score
            return score, alpha, beta

        if maximize:
            move = random.choice(list(board.legal_moves))
            board.push(move)
            # print(self._random_walk_minimax(board, depth + 1, alpha, beta, False))
            value, alpha, beta = self._random_walk_minimax(board, depth + 1, alpha, beta, False)
            board.pop()
            beta = min(beta, value)
        else:
            move = random.choice(list(board.legal_moves))
            board.push(move)
            # print(self._random_walk_minimax(board, depth + 1, alpha, beta, True))
            value, alpha, beta = self._random_walk_minimax(board, depth + 1, alpha, beta, True)
            board.pop()
            alpha = max(alpha, value)
        return value, alpha, beta


    def _minimax(self, board, depth, alpha, beta, maximize):

        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if depth >= self.max_depth:
            score = self.umka.evaluate(board, depth, maximize)
            self.print_info(depth, board)
            self.cache[zobrist_hash(board)] = score
            return score

        if maximize:
            value = INF
            for move in board.legal_moves:
                # if self.time_is_up():
                #     return -INF
                board.push(move)
                value = min(value,
                            self._minimax(board, depth + 1, alpha, beta, False))
                board.pop()

                # if(abs(value) == CHECKMATE):
                #     break

                if value <= alpha:
                    break

                beta = min(beta, value)
        else:
            value = -INF
            for move in board.legal_moves:
                # if self.time_is_up():
                #     return INF
                board.push(move)
                value = max(value,
                            self._minimax(board, depth + 1, alpha, beta, True))
                board.pop()
                #
                # if(abs(value) == CHECKMATE):
                #     break

                if value >= beta:
                    break

                alpha = max(alpha, value)
        return value


    def _fast_ll_minimax(self, board, depth, alpha, beta, maximize):

        self.nodes += 1

        if zobrist_hash(board) in self.cache:
            return self.cache[zobrist_hash(board)]

        if depth == self.max_depth - 1:
            sons = []
            for move in board.legal_moves:
                board.push(move)
                sons.append(deepcopy(board))
                board.pop()
            score = self.umka.multiple_evaluate(sons, depth, maximize)
            self.print_info(depth, board)
            self.cache[zobrist_hash(board)] = score
            return score

        if maximize:
            value = INF
            for move in board.legal_moves:
                # if self.time_is_up():
                #     return -INF
                board.push(move)
                value = min(value,
                            self._minimax(board, depth + 1, alpha, beta, False))
                board.pop()

                # if(abs(value) == CHECKMATE):
                #     break

                if value <= alpha:
                    break

                beta = min(beta, value)
        else:
            value = -INF
            for move in board.legal_moves:
                # if self.time_is_up():
                #     return INF
                board.push(move)
                value = max(value,
                            self._minimax(board, depth + 1, alpha, beta, True))
                board.pop()
                #
                # if(abs(value) == CHECKMATE):
                #     break

                if value >= beta:
                    break

                alpha = max(alpha, value)
        return value

    def alphabeta_minimax_with_rand_walk(self, board):
        depth = 0
        self.nodes = 0
        self.cache = {}

        beta = INF
        best_val = -INF

        # I will try to implement 100 random ways down to last layer
        # in order to get better initial alpha and beta values on
        # very different set of positions.

        # My code starts here
        # TODO: CHECK ALL VALUES FOR ALL ROOT MOVES
        if board.turn:
            values = []
            for i, rm in enumerate(list(self.root_moves)):
                # self.root_moves.remove(rm)
                board.push(rm.move)
                if board.can_claim_draw():
                    value = 0
                else:
                    value, best_val, beta = self._random_walk_minimax(
                        board, depth + 1, best_val, beta, True)
                board.pop()
                rm.value = -value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if self.best_val == INF:
                    break
                self.print_info(depth, board)
                # print('new alpha: {}'.format(best_val))
                # print('new beta: {}'.format(beta))
                values.append(value)
            for rm in list(self.root_moves):
                self.root_moves.remove(rm)
                board.push(rm.move)
                if board.can_claim_draw():
                    value = 0
                else:
                    value = self._minimax(
                        board, depth + 1, best_val, beta, True)
                board.pop()
                if(rm.value > value):
                    print('st rm value {}'.format(rm.value))
                    print('fin rm value {}'.format(value))
                    rm.value = -value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if self.best_val == INF:
                    break

                self.print_info(depth, board)

        else:

            for i, rm in enumerate(list(self.root_moves)):
                board.push(rm.move)
                if board.can_claim_draw():
                    value = 0
                else:
                    value, best_val, beta = self._random_walk_minimax(
                        board, depth + 1, best_val, beta, True)
                board.pop()
                rm.value = value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if self.best_val == -INF:
                    break
                self.print_info(depth, board)
                # print('new alpha: {}'.format(best_val))
                # print('new beta: {}'.format(beta))
            for rm in list(self.root_moves):
                self.root_moves.remove(rm)
                board.push(rm.move)
                if board.can_claim_draw():
                    value = 0
                else:
                    value = self._minimax(
                        board, depth + 1, best_val, beta, False)
                board.pop()
                if (rm.value < value):
                    rm.value = value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if self.best_val == -INF:
                    break
                self.print_info(depth, board)

        # My code ends here


    def alphabeta_minimax(self, board):
        depth = 0
        self.nodes = 0
        self.cache = {}

        beta = INF
        best_val = -INF

        if board.turn:
            for rm in list(self.root_moves):
                self.root_moves.remove(rm)
                board.push(rm.move)
                if board.can_claim_draw():
                    value = 0
                else:
                    value = self._minimax(
                        board, depth + 1, best_val, beta, True)
                board.pop()
                rm.value = -value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if abs(self.best_val) >= CHECKMATE:
                    break
                self.print_info(depth, board)
        else:
            for rm in list(self.root_moves):
                self.root_moves.remove(rm)
                board.push(rm.move)
                if board.can_claim_draw():
                    value = 0
                else:
                    value = self._minimax(
                        board, depth + 1, best_val, beta, False)
                board.pop()
                rm.value = value
                bisect.insort_right(self.root_moves, rm)
                self.best_move = self.root_moves[0].move
                self.best_val = self.root_moves[0].value
                if abs(self.best_val) >= CHECKMATE:
                    break
                self.print_info(depth, board)

    def print_info(self, depth, board):
        if (time.time() - self.last_time_info_printed) < 2:
            return
        move_time = time.time() - self.st
        t = int(move_time) * 1000
        print(
            "info time= %8s depth= %8s nodes= %8s nps= %8s cp= %6s pv= %s %s %s" % (
                t, depth, self.nodes, int(self.nodes / move_time),
                int(1000 * self.best_val),
                self.best_move, self.root_moves,
                str(chess.pgn.Game().from_board(board)).split("\n\n")[1]))

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
            # self.alphabeta_minimax(board)
            self.alphabeta_minimax_with_rand_walk(board)
            d += 1
            print("---------------------->", self.best_move)
            if abs(self.best_val) >= CHECKMATE:
                break
        print('time on move: {}'.format(time.time() - self.st))
        return self.best_move


class SortableMove(Move):
    def __init__(self, move, value=INF):
        self.move = move
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return "%s %s" % (self.move, self.value)