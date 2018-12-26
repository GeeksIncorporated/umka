import os
import torch

PROJECT_ROOT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "core", "models")
PATH_TO_MODEL = os.path.join(MODELS_DIR, "model.pth.tar")
DEPTH = 4
# ENABLE_OPENING_BOOK = False
ENABLE_OPENING_BOOK = True

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Running on", torch.cuda.get_device_name(0))
    print("Capabilities", torch.cuda.get_device_capability(0))
    print("Properties", torch.cuda.get_device_properties(0))
    print("Max mem allocated", torch.cuda.max_memory_allocated(0))
    print("Mac mem cached", torch.cuda.max_memory_cached(0))

else:
    DEVICE = torch.device('cpu')
    print("Running on CPU :(")
