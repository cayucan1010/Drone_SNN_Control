import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from snntorch import utils
import os

# Import your custom modules
from model import GateDetectorSNN
from dataset import AirSimGateDataset
from config import *

def train():
    # 1. Device Setup (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load Dataset
    # Ensure this matches the path where your AirSim script saves data
    dataset_path = "SNN_Gate_Dataset/train" 
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} not found. Run your collection script first!")
        return

    full_dataset = AirSimGateDataset(dataset_path)
    
    # Split Dataset (Using your config ratios)
    n = len(full_dataset)
    train_size = int(n * TRAIN_SPLIT)
    val_size = n - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model, Optimizer, and Loss
    model = GateDetectorSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE)
    
    # MSE works well for regression (Coordinates) and Probability (Confidence)
    criterion = nn.MSELoss()

    print(f"Starting Training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            # x_batch shape from DataLoader: [Batch, T, 1, H, W]
            # SNN needs: [T, Batch, 1, H, W]
            x_batch = x_batch.transpose(0, 1).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            
            # Reset SNN hidden states for the new batch
            utils.reset(model)
            
            y_pred = model(x_batch) # Returns [Batch, 5]
            
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.transpose(0, 1).to(device)
                y_batch = y_batch.to(device)
                
                utils.reset(model)
                y_pred = model(x_batch)
                
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train Loss: {avg_train_loss:.6f} "
              f"| Val Loss: {avg_val_loss:.6f} "
              f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 4. Save the trained weights
    torch.save(model.state_dict(), "gate_detector_snn.pth")
    print("Training complete. Model saved as gate_detector_snn.pth")

if __name__ == "__main__":
    train()