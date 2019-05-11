import torch as torch
import torch.nn as nn

INPUT_SIZE = 20
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001


class SimpleNeuralNet(nn.Module):
    """A Neural Network with a hidden layer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(LEARNING_RATE)

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        return output


nn = SimpleNeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
optimizer = torch.optim.SGD(
    nn.parameters(),
    lr=LEARNING_RATE,
    momentum=0.7,
    nesterov=True)

tensors = [list(map(int, format(1 << (i - 1), '020b'))) for i in range(1, 21)]
t = lambda n: tensors[n - 1]

samples = [[t(1), t(3), t(3)],
           [t(9), t(7), t(4)],
           [t(11), t(14), t(10)],
           [t(20), t(16), t(12)]]

labels = [5, 8, 15, 18]
loss_rate = 100
while loss_rate > 0.001:
    for i, sample in enumerate(samples):
        sample = torch.FloatTensor(sample)
        current = nn(sample)
        expected = torch.FloatTensor([labels[i]] * 3)
        loss = torch.nn.MSELoss()
        delta = loss(current, expected.unsqueeze(1))
        loss_rate = delta.item()
        print("===========================")
        print(labels[i], current.mean().item())
        print("Loss rate:", loss_rate)
        optimizer.zero_grad()
        delta.backward()
        optimizer.step()

s = torch.FloatTensor([t(3), t(6), t(7)]).squeeze()
nn.eval()
val = nn(s)
print("My best guess:", val.mean().item())

