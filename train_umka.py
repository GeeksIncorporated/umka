import csv
from multiprocessing.pool import Pool
from random import randint
import chess
import os
import torch
import torch.nn as nn
from chess.pgn import scan_headers
import sys
sys.setrecursionlimit(3600000)

input_size = 64
hidden_size = 2048
OUTPUT_SIZE = 1
lr = 0.001


class NeuralNet(nn.Module):
    """A Neural Network with 4 a hidden layer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(lr)

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        output = self.relu(output)
        output = self.layer4(output)
        return output


model = NeuralNet(input_size, hidden_size, OUTPUT_SIZE)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.7,
    nesterov=True)

def load_model_if_exists():
    if os.path.exists("model/model.pth.tar"):
        print("=> loading checkpoint")
        checkpoint = torch.load("model/model.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.train()

pieaces = {".": 0,
           "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 10,
           "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -10}


def board_tensor(board):
    res = []
    for c in str(board):
        if c == '\n':
            continue
        elif c == ' ':
            continue
        res.append(pieaces[c] / 10)
    return res


def sample_game(game):
    board = game.board()
    samples = []
    main_line = list(game.main_line())
    for move in main_line:
        if not board.turn:
            board.push(move)
            continue
        position = board_tensor(board=board)
        samples.append(position)
        board.push(move)
    return samples

def load_labels():
    game_labels = {}
    for event_id, position_scores in csv.reader(open("data/pgn/stockfish.csv")):
        game_labels[event_id] = position_scores
    return game_labels


def foo(game):
    try:
        game_labels = load_labels()
        event = game.headers["Event"]
        samples = sample_game(game)
        labels = game_labels[event]
        labels = labels.split(' NA')[0]
        labels = list(map(lambda x: float(x) / 1000, labels.split(' ')[::2]))

        # plt.hist((labels), bins=len(labels)); plt.show()
        # input = torch.from_numpy(np.array(samples[:len(labels)]))

        current = model(torch.FloatTensor(samples)).squeeze()
        expected = torch.FloatTensor(labels)
        loss = nn.MSELoss()
        delta = loss(current, expected)
        optimizer.zero_grad()
        delta.backward()
        optimizer.step()
        print("%.6s,\t%.6s\tLoss: %.6s" % (
            current[0].item(), labels[0], delta.item()))

        if randint(0, 1000) == 999:
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "model/model.pth.tar")
            print("----------")
    except: pass

def train():
    pgn = open("data/pgn/data.pgn")
    load_model_if_exists()
    pool = Pool(4)
    for epoch in range(100):
        games = []
        for i in range(50000):
            game = chess.pgn.read_game(pgn)
            games.append(game)
        pool.map(foo, games)


if __name__ == "__main__":
    train()