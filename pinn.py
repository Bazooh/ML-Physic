from segmentation_models_pytorch import Unet
import torch
import tqdm

BATCH_SIZE = 16
EPOCHS = 10

model = Unet(encoder_name="resnet18", 
             in_channels=1, 
             classes=1)

def train(model, loss, optimizer, train_loader):
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

def pinn_loss(u_grid_label, u_grid_predict, f_grid):

    ddxu = 0
    pass
