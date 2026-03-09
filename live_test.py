import airsim
import torch
import numpy as np
import time
from model import GateDetectorSNN
from snntorch import spikegen, utils
from config import IMG_SIZE, TIMESTEPS

# 1. Setup & Load Model
device = torch.device("cpu") 
model = GateDetectorSNN().to(device)

try:
    model.load_state_dict(torch.load("gate_detector_snn.pth", map_location=device))
    model.eval()
    print("SNN Brain loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. AirSim Connection & Initialization
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
client.moveToZAsync(-3, 2).join() 

# 3. State Variables & Thresholds
gate_count = 0
gate_in_view = False
last_passed_time = 0

# --- CONTROL TUNING ---
MIN_ALTITUDE = 1.5   # Safety floor (m)
MAX_ALTITUDE = 6.0   # Safety ceiling (m)
Y_DEADZONE = 0.05    # Ignore vertical errors smaller than 5%
P_GAIN_XY = 3.5      # Steering sensitivity
P_GAIN_Z = 4.5       # Altitude sensitivity

print(f"Flight active. Floor: {MIN_ALTITUDE}m | Ceiling: {MAX_ALTITUDE}m")

try:
    while True:
        # --- DATA ACQUISITION ---
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        
        # Get altitude (Distance from ground)
        try:
            # Try distance sensor first
            dist_data = client.getDistanceSensorData(distance_sensor_name="Distance")
            current_dist = dist_data.distance
        except:
            # Fallback: Estimated Z (Inverted because -Z is UP)
            current_dist = -client.getMultirotorState().kinematics_estimated.position.z_val

        if not responses or responses[0].width == 0: 
            continue

        # --- PRE-PROCESSING ---
        img_raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
        img_gray = 0.299 * img_raw[:,:,2] + 0.587 * img_raw[:,:,1] + 0.114 * img_raw[:,:,0]
        h, w = img_gray.shape
        img_resized = img_gray[::h//IMG_SIZE, ::w//IMG_SIZE][:IMG_SIZE, :IMG_SIZE]
        
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        spikes = spikegen.rate(img_tensor, num_steps=TIMESTEPS).to(device)

        # --- SNN INFERENCE ---
        utils.reset(model)
        with torch.no_grad():
            output = model(spikes)
            
        pred = output[0].cpu().numpy()
        conf, x_p, y_p, w_p, h_p = map(float, pred)

        # --- FLIGHT CONTROL LOGIC ---
        vx, vy, vz = 0.0, 0.0, 0.0

        if conf > 0.5:
            err_x = x_p - 0.5
            err_y = y_p - 0.5 
            
            vx = float(1.2)  # Forward speed
            vy = float(err_x * P_GAIN_XY)
            
            # Apply Deadzone to Z-axis to prevent "hunting" or infinite climbing
            if abs(err_y) < Y_DEADZONE:
                vz = float(0.0)
            else:
                # Inverted: gate below center (err_y > 0) -> command UP (negative vz)
                vz = float(-err_y * P_GAIN_Z)
            
            gate_in_view = True
            print(f"LOCKED | Conf: {conf:.2f} | Y_pred: {y_p:.2f} | vy: {vy:.2f} | vz: {vz:.2f}")
        else:
            # Pass counting logic
            if gate_in_view and (time.time() - last_passed_time > 2.0):
                gate_count += 1
                last_passed_time = time.time()
                print(f"\n!!! GATE PASSED !!! Count: {gate_count}\n")
                gate_in_view = False
            
            # Search Mode
            client.rotateByYawRateAsync(30, duration=0.1)

        # --- HARD SAFETY OVERRIDES ---
        if current_dist < MIN_ALTITUDE:
            vz = float(-2.0) # Force climb
            print(f"CRITICAL: Low Alt ({current_dist:.1f}m) - Climbing")
        elif current_dist > MAX_ALTITUDE:
            vz = float(2.0)  # Force descend
            print(f"CRITICAL: High Alt ({current_dist:.1f}m) - Descending")

        client.moveByVelocityAsync(vx, vy, vz, duration=0.1)
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nManual Override.")

finally:
    client.hoverAsync().join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Safe landing complete.")