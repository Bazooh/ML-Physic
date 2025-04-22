# ruff: noqa: F403, F405

import random

# from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import derive2_x, derive2_y, f_grid, visu
from constants import *
from torch.utils.data import DataLoader, Dataset
from finite_diff import create_dataset, simulate_u_grid


class GridDataset(Dataset):
    def __init__(self, dataset: dict[tuple[float, float], torch.Tensor]):
        self.dataset = [
            (v.to(DEVICE), f_grid(*k, 64).to(DEVICE)) for k, v in dataset.items()
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_u_f(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the u and f tensors from the dataset.
        :return: tuple of (u, f) tensors
        """
        u = torch.stack([item[0] for item in self.dataset])
        f = torch.stack([item[1] for item in self.dataset])
        return u, f


def train(model: nn.Module, loss, optimizer, train_loader, test_set) -> list[float]:
    model.train()
    losses: list[float] = []
    for epoch in tqdm(range(EPOCHS)):
        pbar = tqdm(train_loader)
        for data in pbar:
            optimizer.zero_grad()
            u_grid_label, f_grid_ = data
            u_grid_predict = model(f_grid_.unsqueeze(1))
            loss_value = loss(u_grid_label, u_grid_predict, f_grid_)
            loss_value.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch: 2}, loss: {loss_value.item(): 10e}")

        u, f = test_set.get_u_f()
        u_pred = model(f.unsqueeze(1))
        test_loss = loss(u, u_pred, f)
        print(f"Test loss: {test_loss.item(): 10e}")
        losses.append(test_loss.item())

    return losses


def pinn_loss(
    u_grid_label: torch.Tensor, u_grid_predict: torch.Tensor, f_grid_: torch.Tensor
):
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


dataset = create_dataset(a_range=(-10, 10), b_range=(-10, 10), size=1000)
train_loader = DataLoader(GridDataset(dataset), batch_size=BATCH_SIZE)
test_dataset = create_dataset(a_range=(-10, 10), b_range=(-10, 10), size=1000)
test_loader = DataLoader(GridDataset(test_dataset), batch_size=BATCH_SIZE)


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.dense = nn.Linear(64 * 64, 64 * 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.flatten(start_dim=1)
        x = self.dense(x)

        return x.view(x.shape[0], 64, 64)


model = Dense().to(DEVICE)
state_dict = model.state_dict()
losses: list[list[float]] = []
for i in range(3):
    model.load_state_dict(state_dict)

    val = [0, 10e-7, 10e-8][i]

    PINN_LOSS_PHYSICS_PROP = val

    model.train()
    losses.append(
        train(
            model=model,
            loss=pinn_loss,
            optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
            train_loader=train_loader,
            test_set=GridDataset(create_dataset(size=100)),
        )
    )
    print("Training complete.")
    torch.save(model.state_dict(), f"{val:6e}.pth")
    model.eval()

    loss = 0
    for batch in test_loader:
        u_grid_label, f_grid_ = batch
        with torch.no_grad():
            u_grid_predict = model(f_grid_.unsqueeze(1))
        loss += nn.MSELoss()(u_grid_label, u_grid_predict).item()

    print(f"Loss {val:6e}.pth: {loss / len(test_loader)}")


import matplotlib.pyplot as plt

# Plot the losses for each training run
for i, loss in enumerate(losses):
    plt.plot(loss, label=f"Run {[0, 10e-7, 10e-8][i]:6e}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.grid(True)
plt.show()
