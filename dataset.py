import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from snntorch import spikegen
from config import IMG_SIZE, TIMESTEPS

class AirSimGateDataset(Dataset):
    def __init__(self, data_path):
        """
        data_path: Path to the folder containing 'images' and 'labels'
        """
        self.img_dir = os.path.join(data_path, "images")
        self.label_dir = os.path.join(data_path, "labels")
        
        # Get list of all image files, ensuring they are sorted
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
        # Standardize image processing
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("L") # Convert to grayscale (1 channel)
        img_tensor = self.transform(image)

        # 2. Convert to Spikes (Rate Coding)
        # SNNs need a temporal sequence [T, C, H, W]
        # We do this here so the DataLoader can parallelize it on the CPU
        spikes = spikegen.rate(img_tensor, num_steps=TIMESTEPS)

        # 3. Load Label and append Confidence Score
        label_path = os.path.join(self.label_dir, img_name.replace(".png", ".txt"))
        
        # Check if label exists (in case some images don't have gates)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().split()
                # We add a '1.0' at the beginning as the "Confidence Score"
                # Label becomes: [Conf=1.0, x, y, w, h]
                label_values = [1.0] + [float(x) for x in line[1:]]
        else:
            # If no label file exists, it's a background image (Conf=0.0)
            label_values = [0.0, 0.5, 0.5, 0.0, 0.0]

        label = torch.tensor(label_values, dtype=torch.float32)

        return spikes, label