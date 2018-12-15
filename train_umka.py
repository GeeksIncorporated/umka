import os
import sys
import torch
from core.umka import Umka
from core.utils import annotated_sample_generator
from settings import MODELS_DIR

sys.setrecursionlimit(3600000)


def train_with_annotated_pgn():
    for epoch in range(100):
        with Umka(path=os.path.join(MODELS_DIR, "model.pth.tar"),
                  training_enabled=True) as umka:
            sample_generator = annotated_sample_generator()
            for samples, labels in sample_generator:
                umka.train(torch.FloatTensor(samples),
                           torch.FloatTensor(labels))


if __name__ == "__main__":
    train_with_annotated_pgn()
