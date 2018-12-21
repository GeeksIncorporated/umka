import os

import torch

PROJECT_ROOT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "core", "models")
PATH_TO_MODEL = os.path.join(MODELS_DIR, "model.pth.tar")
DEPTH = 3

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Running on", torch.cuda.get_device_name(DEVICE))
    print("Capabilities", torch.cuda.get_device_capability(DEVICE))
    print("Properties", torch.cuda.get_device_properties(DEVICE))
    print("Max mem allocated", torch.cuda.max_memory_allocated(DEVICE))
    print("Mac mem cached", torch.cuda.max_memory_cached(DEVICE))

else:
    DEVICE = torch.device('cpu')
    print("Running on CPU :(")
