import datetime
import os
import shutil
import sys
from random import randint
import torch
from chess.polyglot import open_reader

from core.nn import UmkaNeuralNet, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, \
    LEARING_RATE, TrainingDisabledOnModel
from core.utils import board_tensor, show_board, board_material
from settings import DEVICE, ENABLE_OPENING_BOOK, AI_ENABLED, CHECKMATE

piece = {'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000}
pst = {
    'P': (0, 0, 0, 0, 0, 0, 0, 0,
          78, 83, 86, 73, 102, 82, 85, 90,
          7, 29, 21, 44, 40, 31, 44, 7,
          -17, 16, -2, 15, 14, 0, 15, -13,
          -26, 3, 10, 9, 6, 1, 0, -23,
          -22, 9, 5, -11, -10, -2, 3, -19,
          -31, 8, -7, -37, -36, -14, 3, -31,
          0, 0, 0, 0, 0, 0, 0, 0),
    'N': (-66, -53, -75, -75, -10, -55, -58, -70,
          -3, -6, 100, -36, 4, 62, -4, -14,
          10, 67, 1, 74, 73, 27, 62, -2,
          24, 24, 45, 37, 33, 41, 25, 17,
          -1, 5, 31, 21, 22, 35, 2, 0,
          -18, 10, 13, 22, 18, 15, 11, -14,
          -23, -15, 2, 0, 2, 0, -23, -20,
          -74, -23, -26, -24, -19, -35, -22, -69),
    'B': (-59, -78, -82, -76, -23, -107, -37, -50,
          -11, 20, 35, -42, -39, 31, 2, -22,
          -9, 39, -32, 41, 52, -10, 28, -14,
          25, 17, 20, 34, 26, 25, 15, 10,
          13, 10, 17, 23, 17, 16, 0, 7,
          14, 25, 24, 15, 8, 25, 20, 15,
          19, 20, 11, 6, 7, 6, 20, 16,
          -7, 2, -15, -12, -14, -15, -10, -10),
    'R': (35, 29, 33, 4, 37, 33, 56, 50,
          55, 29, 56, 67, 55, 62, 34, 60,
          19, 35, 28, 33, 45, 27, 25, 15,
          0, 5, 16, 13, 18, -4, -9, -6,
          -28, -35, -16, -21, -13, -29, -46, -30,
          -42, -28, -42, -25, -25, -35, -26, -46,
          -53, -38, -31, -26, -29, -43, -44, -53,
          -30, -24, -18, 5, -2, -18, -31, -32),
    'Q': (6, 1, -8, -104, 69, 24, 88, 26,
          14, 32, 60, -10, 20, 76, 57, 24,
          -2, 43, 32, 60, 72, 63, 43, 2,
          1, -16, 22, 17, 25, 20, -13, -6,
          -14, -15, -2, -5, -1, -10, -20, -22,
          -30, -6, -13, -11, -16, -11, -16, -27,
          -36, -18, 0, -19, -15, -15, -21, -38,
          -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (4, 54, 47, -99, -99, 60, 83, -62,
          -32, 10, 55, 56, 56, 55, 10, 3,
          -62, 12, -57, 44, -67, 28, 37, -31,
          -55, 50, 11, -4, -19, 13, 0, -49,
          -55, -43, -52, -28, -51, -47, -8, -50,
          -47, -42, -43, -79, -64, -32, -29, -32,
          -4, 3, -14, -50, -57, -18, 13, 4,
          17, 30, -3, -14, 6, -1, 40, 18),
}


class Umka:
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
                "data/opennings/bookfish.bin") as reader:
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
            # print(' mat sc: {}'.format(material_score))
            # print(' pos sc: {}'.format(position_score))
            score = material_score + position_score
            # score /= float(depth)
        # print(material_score, position_score)
        show_board(board, material_score, position_score)
        return score

    def multiple_evaluate(self, boards, depth, maximize):
        material_scores = []
        position_scores = []
        scores = []

        material_scores.append(10 * board_material(boards[0]))
        position_scores.append(0)
        scores.append(0)
        A = board_tensor(board=boards[0])
        samples = torch.Tensor(A)

        print('i: {}'.format(0))

        for i, board in enumerate(boards):
            if(i!=0):
                print('i: {}'.format(i))
                material_scores.append(10 * board_material(board))
                position_scores.append(0)
                scores.append(0)
                A = board_tensor(board=board)
                samples.add(torch.Tensor(A))
                # print('A: {}'.format(A))
                # print('samples: {}'.format(samples))

        input = torch.Tensor(samples).to(DEVICE)
        evaluation = self.model(input)
        position_scores = evaluation.item()

        print(' sc : {}'.format(scores))
        print(' mat sc: {}'.format(material_scores))
        print(' pos sc: {}'.format(position_scores))

        # self.prev_material_score = int(material_score)
        for i, board in enumerate(boards):
            if board.is_checkmate():
                scores[i] = CHECKMATE
            else:
                if not maximize:
                    position_score = -position_scores

                scores[i] = material_scores[i] + position_scores[i]
                # score /= float(depth)
            # print(material_score, position_score)
            show_board(board, material_scores[i], position_scores[i])
        return scores