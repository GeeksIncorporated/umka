import time


class MiniMaxMDT:
    def __init__(self, umka):
        self.umka = umka
        self.best_move = None
        self.cache = {}

    def min_play(self, board, depth, alpha, beta):
        self.nodes += 1
        if board.is_game_over() or depth <= 0:
            return self.umka.evaluate(board)

        value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = min(value, self.max_play(board, depth - 1, alpha, beta))
            self.best_move = board.pop()
            if value < alpha:
                return value
            beta = min(beta, value)
        return value

    def max_play(self, board, depth, alpha, beta):
        self.nodes += 1
        if board.is_game_over() or depth <= 0:
            return self.umka.evaluate(board)

        value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            value = max(value, self.min_play(board, depth - 1, alpha, beta))
            self.best_move = board.pop()
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def run(self, board, depth):
        self.nodes = 0
        self.st = time.time()
        firstguess = 0
        for d in range(1, depth+1):
            firstguess = self.MTDF(board, firstguess, d)
            # if times_up():
            #     break
        t = time.time() - self.st
        print("time %0.3s" % t, "nodes %0.5s" % self.nodes,
              "nps %0.6s" % (self.nodes / t), "cp %0.4s" % firstguess)
        return self.best_move

    def __run(self, board, depth):
        self.nodes = 0
        self.cache = {}
        self.st = time.time()
        best_val = float('-inf')
        beta = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            value = self.min_play(board, depth - 1, best_val, beta)
            board.pop()
            if value > best_val:
                best_val = value
                best_move = move
        t = time.time() - self.st
        print("time %0.3s" % t, "nodes %0.5s" % self.nodes, "nps %0.6s" % (self.nodes/t), "cp %0.4s" % best_val)
        return best_move

    def MTDF(self, board, f, depth):
        g = f
        upperbound = float('inf')
        lowerbound = float('-inf')
        while lowerbound < upperbound:
            if g == lowerbound:
                beta = g + 0.01
            else:
                beta = g

            g = self.max_play(board, depth - 1, beta, beta)
            if g < beta:
                upperbound = g
            else:
                lowerbound = g
        return g