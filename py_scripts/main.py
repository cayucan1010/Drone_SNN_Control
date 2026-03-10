from connection import init_client
from model import GateDetectorSNN
from controller import fly_through_gates
import torch

def main():
    client = init_client()
    model = GateDetectorSNN()
    model.load_state_dict(torch.load("gate_detector_snn.pth"))
    print("Model loaded.")
    fly_through_gates(client, model, num_gates=3)

if __name__ == "__main__":
    main()