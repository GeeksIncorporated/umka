import os
import torch

PROJECT_ROOT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join('/content', 'drive', 'My Drive')
PATH_TO_MODEL = os.path.join(MODELS_DIR, "model.pth.tar")
print("Path to model", PATH_TO_MODEL)
DEPTH = 4
MAX_DEPTH = 5
CONCURRENTS = 16
ENABLE_OPENING_BOOK = True
AI_ENABLED = True


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
