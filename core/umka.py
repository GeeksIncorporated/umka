import datetime
import os
import shutil
import sys
from random import randint
import matplotlib.pyplot as plt
import torch

from core.nn import UmkaNeuralNet, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, \
    LEARING_RATE, TrainingDisabledOnModel, TrainingEnablingOnModel
from core.utils import board_tensor, show_board
from settings import DEVICE

plt.show()

class Umka:
    def __init__(self, path, training_enabled):
        """
        :param training_enabled: if True set for trainging False - for playing.
        """
        self.path = path
        self.training_enabled = training_enabled
        self.model = UmkaNeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=LEARING_RATE,
            momentum=0.7,
            nesterov=True)

    def __enter__(self):
        self.__load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__save_model()

    def get_model(self):
        return self.model

    def __load_model(self):
        """
        :type path: string specifying path to the tar model file
        for example "model/model.pth.tar"
        :return: Umka model and optimizer
        """
        if os.path.exists(self.path):
            self.__backup_model()
            print("=> loading checkpoint")
            checkpoint = torch.load(self.path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.training_enabled:
                self.model.train()
            else:
                self.model.eval()

    def __backup_model(self):
        checkpoint_name = "%s_bk_%s" % (
            self.path, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        shutil.copy(self.path, checkpoint_name)

    def __save_model(self):
        with open(self.path, "wb+") as f:
            torch.save({
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, f)
        print("saving cehckpoint:", self.path)

    def train(self, samples, labels):
        """
        :param samples: tensor of shape [x, INPUT_SIZE]
        :param labels: tensor of shape [INPUT_SIZE]
        """
        try:
            if not self.training_enabled:
                raise TrainingDisabledOnModel()

            # plt.hist((labels), bins=len(labels)); plt.show()
            # input = torch.from_numpy(np.array(samples[:len(labels)]))

            current = self.model(torch.FloatTensor(samples).to(DEVICE)).squeeze()
            expected = torch.FloatTensor(labels).to(DEVICE)
            loss = torch.nn.MSELoss()
            delta = loss(current, expected)
            self.optimizer.zero_grad()
            delta.backward()
            self.optimizer.step()

            print("%.6s,\t%.6s\tLoss: %.6s" % (
                current[0].item(), labels[0], delta.item()))

            if randint(0, 100) == 99:
                self.__save_model()
                print("----------")
        except:
            print(sys.exc_info())

    def evaluate(self, board):
        if self.training_enabled:
            raise TrainingEnablingOnModel()
        show_board(board)
        sample = board_tensor(board=board)
        input = torch.FloatTensor(sample).to(DEVICE)
        evaluation = self.model(input)
        score = sum(sample) / 10 + evaluation.item()
        if board.is_checkmate():
            score = 100
        if board.turn:
            score = -score
        return score

