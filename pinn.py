# ruff: noqa: F403, F405

from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import derive_x, derive_y, f_grid
from constants import *
from torch.utils.data import DataLoader, Dataset
from finite_diff import create_dataset

class GridDataset(Dataset):
    def __init__(self, dataset: dict[tuple[float, float], torch.Tensor]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key = list(self.dataset.keys())[idx]
        a, b = key
        u_grid_label = self.dataset[key]
        N = u_grid_label.shape[0]
        f_grid_ = f_grid(a, b, N)

        return f_grid_, u_grid_label
    
dataset = create_dataset()
train_loader = DataLoader(
    GridDataset(dataset), batch_size=BATCH_SIZE, shuffle=True
)

model = Unet(encoder_name="resnet18", in_channels=1, classes=1)


def train(model, loss, optimizer, train_loader) -> None:
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            f_grid_, u_grid_label = data
            u_grid_predict = model(f_grid_)
            loss_value = loss(u_grid_label, u_grid_predict, f_grid_)
            loss_value.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Train loss: {loss_value.item()}")


def pinn_loss(
    u_grid_label: torch.Tensor, u_grid_predict: torch.Tensor, f_grid_: torch.Tensor
) -> float:
    data_term = PINN_LOSS_DATA_PROP * nn.MSELoss()(u_grid_label, u_grid_predict)

    ddxu = derive_x(derive_x(u_grid_predict))
    ddyu = derive_y(derive_y(u_grid_predict))

    physics_term = PINN_LOSS_PHYSICS_PROP * nn.MSELoss()(ddxu + ddyu, -f_grid_)

    true_borders = torch.zeros(4 * (u_grid_predict.shape[0] - 1), dtype=torch.float32)
    predicted_borders = torch.cat(
        [
            u_grid_predict[0, :],
            u_grid_predict[-1, :],
            u_grid_predict[1:-1, 0],
            u_grid_predict[1:-1, -1],
        ]
    )
    border_term = PINN_LOSS_BORDER_PROP * nn.MSELoss()(true_borders, predicted_borders)

    return data_term + physics_term + border_term

train(
    model=model,
    loss=pinn_loss,
    optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
    train_loader=train_loader,
)

print("Training complete.")