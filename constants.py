import torch

BATCH_SIZE = 16
EPOCHS = 20

PINN_LOSS_DATA_PROP = 1
PINN_LOSS_PHYSICS_PROP = 1e-8
PINN_LOSS_BORDER_PROP = 0

LEARNING_RATE = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
# DEVICE = "cpu"
