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
from settings import DEVICE, ENABLE_OPENING_BOOK, AI_ENABLED


class Umka:
    def __init__(self, path, training_enabled):
        """
        :param training_enabled: if True set for trainging False - for playing.
        """
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

    def evaluate(self, board, is_capture=False, is_castling=False, evaluate_position=True):
        material_score = board_material(board)
        if AI_ENABLED and not is_capture and evaluate_position:
            sample = board_tensor(board=board)
            input = torch.FloatTensor(sample).to(DEVICE)
            evaluation = self.model(input)
            position_score = evaluation.item()
        elif is_castling:
            position_score = 0.5
        else:
            position_score = 0

        score = material_score + position_score
        show_board(board, material_score, position_score)
        if board.is_checkmate():
            score = 100
        return score

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = PST[p][j] - PST[p][i]
        # Capture
        if q.islower():
            score += PST[q.upper()][119 - j]
        # Castling check detection
        if abs(j - self.kp) < 2:
            score += PST['K'][119 - j]
        # Castling
        if p == 'K' and abs(i - j) == 2:
            score += PST['R'][(i + j) // 2]
            score -= PST['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += PST['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += PST['P'][119 - (j + S)]
        return score
