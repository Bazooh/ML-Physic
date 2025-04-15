from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import derive_x, derive_y
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
    data_term = PINN_LOSS_DATA_PROP * nn.MSELoss()(u_grid_label, u_grid_predict)

    ddxu = derive_x(derive_x(u_grid_predict))
    ddyu = derive_y(derive_y(u_grid_predict))

    physics_term = PINN_LOSS_PHYSICS_PROP * nn.MSELoss()(ddxu + ddyu, -f_grid)

    true_corners = torch.zeros(2,2)
    predicted_corners = torch.tensor([[u_grid_predict[0,0], u_grid_predict[0,-1]],
                                       [u_grid_predict[-1,0], u_grid_predict[-1,-1]]])
    border_term = PINN_LOSS_BORDER_PROP * nn.MSELoss()(true_corners, predicted_corners)

    return data_term + physics_term + border_term

