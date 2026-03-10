# Simulation
IMG_SIZE = 64
TIMESTEPS = 25
IMG_CHANNELS = 1  # grayscale

# SNN / LIF
BETA = 0.95       # membrane decay rate
THRESHOLD = 1.0   # firing threshold

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 25
LR_PATIENCE = 3   # halve LR if val loss stalls for N epochs
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Dataset
NUM_IMAGES = 1000

# Controller
KP = 0.5          # proportional gain, tune empirically
SUCCESS_THRESHOLD = 0.5  # meters from gate center