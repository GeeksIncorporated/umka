import unittest

from chess import Board


class MyTestCase(unittest.TestCase):
    def test_something(self):
        board = Board()
        print(str(board))
        print(board.legal_moves)


if __name__ == '__main__':
    unittest.main()
