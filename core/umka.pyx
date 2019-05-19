# cython: language_level=3

import datetime
import os
import shutil
import sys
from random import randint
import torch
from chess.polyglot import open_reader

from core.nn import UmkaNeuralNet, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, \
    LEARNING_RATE, TrainingDisabledOnModel
from core.utils import board_tensor, show_board, board_material
from settings import DEVICE, ENABLE_OPENING_BOOK, AI_ENABLED, CHECKMATE, \
    PROJECT_ROOT_DIR


cdef class Umka:
    cdef float prev_material_score
    cdef str path
    cdef bint training_enabled
    cdef object model
    cdef object optimizer

    def __init__(self, path, training_enabled):
        """
        :param training_enabled: if True set for trainging False - for playing.
        """
        self.prev_material_score = 0
        self.path = path
        self.training_enabled = training_enabled
        self.model = UmkaNeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(
            DEVICE)
        self.model.share_memory()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=LEARNING_RATE,
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
        for example "models/model.pth.tar"
        :return: Umka model and optimizer
        """

        if os.path.exists(self.path):
            self.__backup_model()
            print("=> loading checkpoint", self.path)
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
        with open(self.path, "wb") as f:
            torch.save({
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, f)
        print("saving cehckpoint:", self.path)

    def train(self, i, samples, labels):
        """
        :param samples: tensor of shape [x, INPUT_SIZE]
        :param labels: tensor of shape [INPUT_SIZE]
        """
        try:
            if not self.training_enabled:
                raise TrainingDisabledOnModel()

            # plt.hist((labels), bins=len(labels)); plt.show()
            # input = torch.from_numpy(np.array(samples[:len(labels)]))

            current = self.model(
                torch.FloatTensor(samples).to(DEVICE)).squeeze()
            expected = torch.FloatTensor(labels).to(DEVICE)
            loss = torch.nn.MSELoss()
            delta = loss(current, expected)
            self.optimizer.zero_grad()
            delta.backward()
            self.optimizer.step()

            print(i, "%.6s,\t%.6s\tLoss: %.6s" % (
                current[0].item(), labels[0], delta.item()))

            if randint(0, 1000) == 999:
                self.__save_model()
                print("-----SAVED-----")
        except:
            print(sys.exc_info())

    def get_move_from_opennings(self, board):
        if not ENABLE_OPENING_BOOK:
            return
        with open_reader(
                os.path.join(
                    PROJECT_ROOT_DIR,
                    "data/opennings/bookfish.bin")) as reader:
            try:
                entry = reader.choice(board)
                print(entry.move(), entry.weight, entry.learn)
                return entry.move()
            except:
                return None

    def evaluate(self, board, depth, maximize):
        material_score = 10 * board_material(board)

        if AI_ENABLED:
            sample = board_tensor(board=board)
            input = torch.FloatTensor(sample).to(DEVICE)
            evaluation = self.model(input)
            position_score = evaluation.item()
        else:
            position_score = 0
        # self.prev_material_score = int(material_score)
        if board.is_checkmate():
            score = CHECKMATE
        else:
            if not maximize:
                position_score = -position_score
            score = material_score + position_score
            # score /= float(depth)
        # print(material_score, position_score)
        show_board(board, material_score, position_score)
        return score

    def evaluate_bulk(self, boards, depth, maximize):
        material_scores = []
        for board in boards:
            material_scores.append(
                10 * board_material(board)
            )

        if AI_ENABLED:
            samples = []
            for board in boards:
                sample = board_tensor(board=board)
                samples.append(sample)
            if not samples:
                return 0
            input = torch.FloatTensor(samples).to(DEVICE)
            position_score = self.model(input)
        else:
            position_score = torch.FloatTensor([0])

        if board.is_checkmate():
            score = CHECKMATE
        else:

            if maximize:
                score = max(material_scores) + position_score.max().item()
            else:
                score = min(material_scores) - position_score.min().item()
            # score /= float(depth)
        show_board(board, score, position_score)
        return score
