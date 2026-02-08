# train.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

from torch.amp import GradScaler, autocast

from model import *
from loss_function import choose_loss_function


# Definiere das Gerät (CPU oder GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(image_dir, density_dir, model_save_path, checkpoint_path, norm_factor, batch_size, num_epochs, patience, lr, loss_function, resume_training): 
    transform = transforms.Compose([
        transforms.Lambda(to_tensor)
    ])

    dataset = DensityMapLuminanceDataset(image_dir, density_dir, norm_factor)
    model = UNET()

    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=8,  
        pin_memory=True  
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)


    scaler = GradScaler()

    loss_array = []
    lr_array = []


    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0


    # Load checkpoint if resuming training
    if resume_training == True:
        
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        loss_array[:start_epoch] = checkpoint['loss_array']
        lr_array[:start_epoch] = checkpoint['lr_array']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        batch_num = 0
        total_batches = len(train_loader)
        for image, density in train_loader:
            image, density = image.to(device), density.to(device)
            batch_num += 1

            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast("cuda"):
                outputs = model(image)
                outputs = torch.nn.functional.interpolate(
                    outputs, size=density.shape[2:], mode='bilinear', align_corners=False
                )
                loss = choose_loss_function(outputs, density, loss_function)

            # Backward Pass with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Updates the scaler for next iteration

            running_loss += loss.item() * image.size(0)
            torch.cuda.empty_cache()

            # Anzeige des Fortschritts der aktuellen Epoche
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_num}/{total_batches}], Loss: {loss.item():.3e}", end='\r')

        print("")
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs} finished')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3e}')
        loss_array.append(epoch_loss)
        lr_array.append(optimizer.param_groups[0]['lr'])

        if epoch > 0:
            loss_progression = loss_array[epoch-1] - loss_array[epoch]
            print(f"Changes in Loss from epoch {epoch} to epoch {epoch+1}: {loss_progression:.3e}")
        formatted_loss_array = [f"{loss:.3e}" for loss in loss_array]
        print(f"Current loss progression: ", formatted_loss_array)
        print("")

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'loss_array': loss_array,
            'lr_array': lr_array,
            'epochs_no_improve': epochs_no_improve}, model_save_path)  # Save the best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1 # Save the best model


        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        model_save_dir = os.path.dirname(model_save_path)
        np.save(f'{model_save_dir}/loss_array.npy', loss_array)  # Loss-Werte in Datei speichern
        np.save(f'{model_save_dir}/lr_array.npy', lr_array)      # Lernraten in Datei speichern

        torch.cuda.empty_cache()