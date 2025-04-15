# ruff: noqa: F403, F405

from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import derive2_x, derive2_y, f_grid
from constants import *
from torch.utils.data import DataLoader, Dataset
from finite_diff import create_dataset


class GridDataset(Dataset):
    def __init__(self, dataset: dict[tuple[float, float], torch.Tensor]):
        N = dataset[list(dataset.keys())[0]][1].shape[0]
        self.dataset = [
            (v.to(DEVICE), f_grid(*k, N).to(DEVICE)) for k, v in dataset.items()
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


dataset = create_dataset(size=1000)
train_loader = DataLoader(GridDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)

model = Unet(encoder_name="resnet18", in_channels=1, classes=1).to(DEVICE)


def train(model: Unet, loss, optimizer, train_loader) -> None:
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        pbar = tqdm(train_loader)
        for data in pbar:
            optimizer.zero_grad()
            f_grid_, u_grid_label = data
            u_grid_predict = model(f_grid_.unsqueeze(1)).squeeze(1)
            loss_value = loss(u_grid_label, u_grid_predict, f_grid_)
            loss_value.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}, Train loss: {loss_value.item()}")


def pinn_loss(
    u_grid_label: torch.Tensor, u_grid_predict: torch.Tensor, f_grid_: torch.Tensor
) -> float:
    data_term = PINN_LOSS_DATA_PROP * nn.MSELoss()(u_grid_label, u_grid_predict)

    ddxu = torch.stack(
        [derive2_x(u_grid_predict[i]) for i in range(u_grid_predict.shape[0])]
    )
    ddyu = torch.stack(
        [derive2_y(u_grid_predict[i]) for i in range(u_grid_predict.shape[0])]
    )

    physics_term = PINN_LOSS_PHYSICS_PROP * nn.MSELoss()(ddxu + ddyu, -f_grid_)

    true_borders = torch.zeros(
        (u_grid_predict.shape[0], 4 * (u_grid_predict.shape[1] - 1)),
        dtype=torch.float32,
        device=DEVICE,
    )
    predicted_borders = torch.cat(
        [
            u_grid_predict[:, 0, :],
            u_grid_predict[:, -1, :],
            u_grid_predict[:, 1:-1, 0],
            u_grid_predict[:, 1:-1, -1],
        ],
        dim=1,
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
