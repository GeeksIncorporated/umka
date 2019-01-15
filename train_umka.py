import sys
from core.__umka import Umka
from core.__utils import annotated_sample_generator
from settings import PATH_TO_MODEL
sys.setrecursionlimit(3600000)


def train_with_annotated_pgn():
    i = 0
    with Umka(path=PATH_TO_MODEL,
              training_enabled=True) as umka:
        for epoch in range(100):
            sample_generator = annotated_sample_generator()
            batch_samples = []
            batch_labels = []
            for samples, labels in sample_generator:
                i += 1
                batch_samples += samples
                batch_labels += labels
                if i % 10 == 9:
                    umka.train(i, batch_samples, batch_labels)
                    batch_labels = []
                    batch_samples = []

if __name__ == "__main__":
    train_with_annotated_pgn()
