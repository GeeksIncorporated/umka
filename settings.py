import os
PROJECT_ROOT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "core", "models")
PATH_TO_MODEL = os.path.join(MODELS_DIR, "model.pth.tar")
DEPTH = 3
DEVICE = 'cpu'