import time


class MiniMax:
    def __init__(self, umka):
        self.umka = umka

    def min_play(self, board, depth, alpha, beta):
        self.nodes += 1
        if board.is_game_over() or depth <= 0:
            return self.umka.evaluate(board)

        value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = min(value, self.max_play(board, depth - 1, alpha, beta))
            board.pop()
            if value <= alpha:
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
            board.pop()
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def run(self, board, depth):
        self.nodes = 0
        self.st = time.time()
        best_val = float('-inf')
        beta = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            value = self.min_play(board, depth - 1, best_val, beta)
            board.pop()
            if value >= best_val:
                best_val = value
                best_move = move
        print(self.nodes, self.nodes/(time.time() - self.st))
        return best_move
