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


def train(model: nn.Module, loss, optimizer, train_loader, test_set) -> None:
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        pbar = tqdm(train_loader)
        for data in pbar:
            optimizer.zero_grad()
            u_grid_label, f_grid_ = data
            u_grid_predict = model(f_grid_.unsqueeze(1))
            loss_value = loss(u_grid_label, u_grid_predict, f_grid_)
            loss_value.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch: 2}, loss: {loss_value.item(): 15.5f}")

        u, f = test_set.get_u_f()
        u_pred = model(f.unsqueeze(1))
        test_loss = loss(u, u_pred, f)
        print(f"Test loss: {test_loss.item(): 10e}")


def pinn_loss(
    u_grid_label: torch.Tensor, u_grid_predict: torch.Tensor, f_grid_: torch.Tensor
) -> float:
    data_term = 1 * nn.MSELoss()(u_grid_label, u_grid_predict)

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

    return data_term


#dataset = create_dataset(size=1000)
#train_loader = DataLoader(GridDataset(dataset), batch_size=BATCH_SIZE)


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

        x = x.view(x.shape[0], 64, 64)

        # put zeros in the borders
        x[:, 0, :] = 0
        x[:, -1, :] = 0
        x[:, :, 0] = 0
        x[:, :, -1] = 0

        return x


model = Dense().to(DEVICE)

#train(
#    model=model,
#    loss=pinn_loss,
#    optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
#    train_loader=train_loader,
#    test_set=GridDataset(create_dataset(size=100)),
#)

#print("Training complete.")
#torch.save(model.state_dict(), "model.pth")


model.load_state_dict(torch.load("models/0.000000e+00.pth"))
model.eval()

"""
a, b, n = random.uniform(-1, 1), random.uniform(-1, 1), 64
new_f = f_grid(a, b, n).unsqueeze(0).unsqueeze(0).to(DEVICE)

u = model(new_f).squeeze(0).cpu().detach()
real_u = simulate_u_grid(a, b, n)

print(nn.MSELoss()(u, real_u).item())

physics_term = PINN_LOSS_PHYSICS_PROP * nn.MSELoss()(
    derive2_x(u) + derive2_y(u), -new_f.squeeze(0).cpu().detach()
)

print(physics_term.item())

visu(u)
visu(real_u)
visu(u - real_u)
"""

test_dataset = create_dataset(a_range=(-10, 10), b_range=(-10, 10), size=1000)
test_loader = DataLoader(GridDataset(test_dataset), batch_size=BATCH_SIZE)

loss = 0
for batch in test_loader:
    u_grid_label, f_grid_ = batch
    with torch.no_grad():
        u_grid_predict = model(f_grid_.unsqueeze(1))
    loss += nn.MSELoss()(u_grid_label, u_grid_predict).item()

print(f"Loss : {loss / len(test_loader)}")
