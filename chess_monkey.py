import csv
import signal
import sys

import chess
import numpy as np
import torch
import torch.nn as nn
from chess.pgn import scan_headers
from torch.autograd import Variable


def signal_handler(sig=None, frame=None):
    print('Saving model')
    torch.save(model.state_dict(), "model.pkl")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

input_size = 64
hidden_size = 2048
OUTPUT_SIZE = 1
lr = 0.005


class NeuralNet(nn.Module):
    """A Neural Network with a hidden layer"""

    # TODO: check conv
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(lr)

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        output = self.relu(output)
        output = self.layer4(output)
        output = self.relu(output)
        output = self.layer5(output)
        return output


model = NeuralNet(input_size, hidden_size, OUTPUT_SIZE)
lossFunction = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.7,
    nesterov=True)

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


i = 0
lsc = 0
game_labels = {}
for event_id, position_scores in csv.reader(open("data/pgn/stockfish.csv")):
    game_labels[event_id] = position_scores

pgn = open("data/pgn/data.pgn")

all_offsets = list(
    offset for offset, headers in scan_headers(pgn))


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


games = []


for epoche in range(100):
    for i, offset in enumerate(all_offsets):
        try:
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            event = game.headers["Event"]
            samples = sample_game(game)
            labels = game_labels[event]
            labels = labels.split(' NA')[0]
            labels = list(map(lambda x: float(x)/1000, labels.split(' ')[::2]))
            # plt.hist((labels), bins=len(labels)); plt.show()
            # input = torch.from_numpy(np.array(samples[:len(labels)]))
            input = torch.from_numpy(np.array(samples))
            current = model(input.float())
            current = current.squeeze()
            expected = torch.from_numpy(np.array(labels))
            loss = nn.MSELoss()
            output = loss(current, Variable(expected).float())
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            lsc = 0.9 * lsc + 0.1*output.item()
            # print(samples[i])
            # print(labels[i])
            # print(epoche, i, "%.6s,\t%s\tLoss: %s" % (current.item(), labels[i], lsc))
            print(epoche, i, "%.6s,\t%s\tLoss: %s" % (current[-1].item(), labels[-1], lsc))
            i += 1
        except:
            print(sys.exc_info())

signal_handler()

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28 * 28)
#         out = model(images)
#         _, predicted = torch.max(out.data, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         print('Accuracy of the network on the 10000 '
#               'test images: {} %'.format(100 * correct / total))
