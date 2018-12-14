import os
from multiprocessing.dummy import Pool

import chess
import torch
from chess.pgn import scan_headers
from chess.uci import popen_engine
import torch.nn as nn
import matplotlib.pyplot as plt

from settings import PROJECT_ROOT_DIR


class NoModelLoaded(Exception): pass


class Engine:

    def __init__(self, name):
        self.name = name
        self.movetime = 50  # ms

    def evaluate(self, board):
        engine = popen_engine("stockfish")
        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)
        engine.position(board)
        engine.go(movetime=self.movetime)
        sc, m = info_handler.info["score"][1]
        engine.quit()
        engine.terminate()
        engine.kill()
        if m: sc = 10000
        score = max([-10, min([10, sc / 1000])])
        return score


class Coach:
    def __init__(self, model):
        self.counter = 0
        self.model = model
        self.engine = Engine('stockfish')
        self.learning_rate = 0.001
        self._path = os.path.join(PROJECT_ROOT_DIR, 'model', 'model.pth.tar')

    def _train(self, samples, labels):
        input = torch.FloatTensor(samples)
        expected = torch.FloatTensor(labels)
        current = self.model(input)
        current = current.squeeze()
        loss = nn.MSELoss()
        delta = loss(current, expected)
        # plt.hist((labels), bins=len(labels));plt.show()
        self.optimizer.zero_grad()
        delta.backward()
        self.optimizer.step()
        print(current, "Curr: %s\tExp: %s\tLoss: %s" % (
            current.mean().item(), expected.mean().item(), delta.item()))
        if self.counter % 10 == 0:
            self._save_model()
        self.counter += 1

    def train_board(self, board):
        if not self.model:
            raise NoModelLoaded()
        sample = board_tensor(board=board)
        label = self.engine.evaluate(board)
        return self._train(sample, label)

    def train_game(self, game):
        if not self.model:
            raise NoModelLoaded()
        labels = []
        samples = []
        board = game.board()
        for move in game.main_line():
            # if board.turn:
            score = self.engine.evaluate(board)
            score = score if board.turn else -1 * score
            labels.append(score)
            position = board_tensor(board=board)
            samples.append(position)
            board.push(move)
        return self._train(samples, labels)

    def train_pgn(self, pgn):
        pgn = open(pgn)
        if not self.model:
            raise NoModelLoaded()
        counter = 0
        # pool = Pool(os.cpu_count())
        games = []
        while True:
            try:
                counter += 1
                game = chess.pgn.read_game(pgn)
                if not game:
                    break
                games.append(game)
            except:
                pass
        # pool.map(self.train_game, games)
        for game in games:
            self.train_game(game)

    def _load_model(self):
        # device = torch.device('cpu')
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.7,
            nesterov=True)

        if os.path.exists(self._path):
            print("=> loading checkpoint")
            checkpoint = torch.load("model/model.pth.tar")
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.eval()

    def _save_model(self):
        return
        if not self.model:
            raise NoModelLoaded()
        print("===== Saving Model ========")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, self._path)

        print("---------------------------")

    def get_model(self):
        return self.model

    def __enter__(self):
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save_model()
        return True


PIEACES = {".": 0,
           "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 10,
           "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -10}


def board_tensor(board):
    res = []
    for c in str(board):
        if c == '\n':
            continue
        elif c == ' ':
            continue
        res.append(PIEACES[c] / 10)
    return res
