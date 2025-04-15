from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn
import tqdm
from utils import derive_x, derive_y, pinn_loss
from constants import *

model = Unet(encoder_name="resnet18", 
             in_channels=1, 
             classes=1)

def train(model, loss, optimizer, train_loader) -> None:
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            f_grid, u_grid_label = data
            u_grid_predict = model(f_grid)
            loss_value = loss(u_grid_label, u_grid_predict, f_grid)
            loss_value.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Train loss: {loss_value.item()}")

def pinn_loss(u_grid_label: torch.Tensor, u_grid_predict: torch.Tensor, f_grid: torch.Tensor) -> float:
    data_term = PINN_ * nn.MSELoss()(u_grid_label, u_grid_predict)

    ddxu = derive_x(derive_x(u_grid_predict))
    ddyu = derive_y(derive_y(u_grid_predict))

    physics_term = nn.MSELoss()(ddxu + ddyu, -f_grid)

