import time

import chess
from lupa import LuaRuntime
from core.umka import Umka
from play import play


class MiniMaxLua:
    def __init__(self, umka):
        self.umka = umka
        self.lua = LuaRuntime(unpack_returned_tuples=True)

    def run(self, board, depth):
        self.nodes = 0
        self.st = time.time()
        best_val = float('-inf')
        beta = float('inf')
        best_move = None
        minimax = self.lua.eval("""
        local function minimax(board, depth, maximize, bestScore)
            if board.is_game_over() or depth <= 0 then
                return tree:heuristic(node)
            end
            local children = tree:children(node)
            if maximize then
                bestScore = -math.huge
                for i, child in ipairs(children) do
                    bestScore = math.max(bestScore, minimax(tree, child, depth - 1, false))
                end
                return bestScore
            else
                bestScore = math.huge
                for i, child in ipairs(children) do
                    bestScore = math.min(bestScore, minimax(tree, child, depth - 1, true))
                end
                return bestScore
            end
        end""")
        return minimax(self.umka.evaluate, board)


if __name__ == "__main__":
    # umka = Umka(path="model/model.pth.tar", training_enabled=False)
    # brain = MiniMaxLua(umka)
    # play(brain)
    board = chess.Board()

    st = time.time()
    lua = LuaRuntime(unpack_returned_tuples=True)
    res = []

    is_over = lua.eval("""
    function (board) 
        for i=1,1000000 do
            board.push_uci('e2e4')
            board.pop()
        end
    end""")
    is_over(board)
    print(1000000.0 / (time.time() - st))

    st = time.time()
    lua = LuaRuntime(unpack_returned_tuples=True)
    res = []
    for i in range(1000000):
        board.push_uci("e2e4")
        board.pop()

    print(1000000.0/(time.time() - st))