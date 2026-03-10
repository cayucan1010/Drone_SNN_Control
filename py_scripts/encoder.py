import numpy as np
import torch
import airsim
from PIL import Image
from config import IMG_SIZE, TIMESTEPS

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0

def rate_encode(frame_norm: np.ndarray, timesteps: int = TIMESTEPS) -> torch.Tensor:
    spikes = (np.random.rand(timesteps, *frame_norm.shape) < frame_norm)
    spikes = torch.tensor(spikes, dtype=torch.float32)  # [T, H, W]
    return spikes.unsqueeze(1)  # [T, 1, H, W]

def get_frame_spikes(client, timesteps=TIMESTEPS) -> torch.Tensor:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img = img.reshape(responses[0].height, responses[0].width, 3)
    frame_norm = preprocess_frame(img)
    return rate_encode(frame_norm, timesteps)