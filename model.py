import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from config import BETA, THRESHOLD

class GateDetectorSNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()

        # Feature Extraction
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, padding=2)
        self.lif1  = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.lif2  = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        # Fully Connected Layers
        self.fc1   = nn.Linear(24 * 16 * 16, 128)
        self.lif3  = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad)

        # Output Layer: 5 neurons [Conf, X, Y, W, H]
        self.fc2   = nn.Linear(128, 5) 
        self.lif4  = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad)

    def forward(self, x):
        T = x.shape[0]
        batch = x.shape[1]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        mem4_sum = torch.zeros(batch, 5, device=x.device)

        for t in range(T):
            c1 = self.conv1(x[t])
            spk1, mem1 = self.lif1(c1, mem1)
            p1 = self.pool1(spk1)

            c2 = self.conv2(p1)
            spk2, mem2 = self.lif2(c2, mem2)
            p2 = self.pool2(spk2)

            flat = p2.view(batch, -1)
            spk3, mem3 = self.lif3(self.fc1(flat), mem3)

            # Accumulate membrane potential for the 5 output neurons
            _, mem4 = self.lif4(self.fc2(spk3), mem4)
            mem4_sum += mem4

        # Average and squash to [0, 1] range
        y_pred = torch.sigmoid(mem4_sum / T) 
        return y_pred