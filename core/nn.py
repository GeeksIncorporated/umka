import torch.nn as nn

INPUT_SIZE = 64
HIDDEN_SIZE = 2048
OUTPUT_SIZE = 1
LEARING_RATE = 0.001


class TrainingDisabledOnModel():
    pass


class TrainingEnablingOnModel():
    pass


class UmkaNeuralNet(nn.Module):
    """A Neural Network with a hidden layer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(UmkaNeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(LEARING_RATE)

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        output = self.relu(output)
        output = self.layer4(output)
        return output
