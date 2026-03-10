import torch
import numpy as np
import airsim
from encoder import get_frame_spikes
from config import KP, TIMESTEPS

def predict_gate(model, client) -> np.ndarray:
    spikes = get_frame_spikes(client, TIMESTEPS)  # [T, 1, H, W]
    spikes = spikes.unsqueeze(1)                   # [T, 1, 1, H, W]
    model.eval()
    with torch.no_grad():
        bbox = model(spikes)
    return bbox.squeeze().numpy()

def compute_velocity(bbox: np.ndarray):
    xcenter, ycenter = bbox[0], bbox[1]
    vx = KP * (xcenter - 0.5)
    vy = KP * (ycenter - 0.5)
    return float(vx), float(vy), 0.0

def fly_through_gates(client, model, num_gates=3):
    gates_cleared = 0

    while gates_cleared < num_gates:
        bbox = predict_gate(model, client)
        vx, vy, vz = compute_velocity(bbox)

        client.moveByVelocityAsync(
            vx, vy, vz,
            duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0)
        ).join()

        if gate_cleared(client):
            gates_cleared += 1
            print(f"Gate {gates_cleared} cleared!")

    print(f"Course complete: {gates_cleared}/{num_gates} gates cleared.")

def gate_cleared(client) -> bool:
    """Placeholder: implement using drone position vs known gate positions."""
    return False