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
    print("--- SNN RACING BRAIN ONLINE ---")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. AirSim Connection
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
client.moveToZAsync(-3, 2).join() 

# 3. Racing State Variables
gate_count = 0
gate_in_view = False
last_passed_time = 0
TARGET_ALTITUDE = 3.0  # The height the drone tries to maintain when not diving/climbing for a gate

# --- RACING & SAFETY TUNING ---
BASE_SPEED = 4.5      
P_GAIN_YAW = 85.0     
P_GAIN_Z = 5.5        
Y_DEADZONE = 0.05

# SAFETY LIMITS
MIN_ALTITUDE = 1.5    
MAX_ALTITUDE = 8.0    

try:
    while True:
        # --- 1. SENSE: Image & Altitude Data ---
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        
        # Get Current Altitude (Distance from ground)
        try:
            dist_data = client.getDistanceSensorData(distance_sensor_name="Distance")
            current_dist = dist_data.distance
        except:
            current_dist = -client.getMultirotorState().kinematics_estimated.position.z_val

        if not responses or responses[0].width == 0: 
            continue

        # --- 2. PROCESS: SNN Inference ---
        img_raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
        img_gray = 0.299 * img_raw[:,:,2] + 0.587 * img_raw[:,:,1] + 0.114 * img_raw[:,:,0]
        h, w = img_gray.shape
        img_resized = img_gray[::h//IMG_SIZE, ::w//IMG_SIZE][:IMG_SIZE, :IMG_SIZE]
        
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        spikes = spikegen.rate(img_tensor, num_steps=TIMESTEPS).to(device)

        utils.reset(model)
        with torch.no_grad():
            output = model(spikes)
            
        pred = output[0].cpu().numpy()
        conf, x_p, y_p, w_p, h_p = map(float, pred)

        # --- 3. BRAIN TELEMETRY ---
        bar = '█' * int(round(20 * conf)) + '-' * (20 - int(round(20 * conf)))
        off_x = x_p - 0.5
        off_y = y_p - 0.5

        # --- 4. ACT: Racing Flight Control with Altitude Correction ---
        vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

        if conf > 0.5:
            # RACING MODE
            vx = float(BASE_SPEED)
            yaw_rate = float(off_x * P_GAIN_YAW)
            
            # Combine SNN target with a baseline altitude hold
            # If the SNN sees a gate, it steers towards it. 
            # If the gate is perfectly centered, vz will stay 0.0 (maintaining current height)
            vz = float(off_y * P_GAIN_Z)
            
            gate_in_view = True
            status = f"LOCKED | Alt: {current_dist:.1f}m"
        else:
            # SEARCH MODE + AUTOMATIC ALTITUDE HOLD
            yaw_rate = 60.0
            vx = 0.0
            
            # If we lose the gate, fight to get back to TARGET_ALTITUDE (3m)
            # altitude_error = (current_height - target_height)
            # Since -Z is UP, a positive error means we are too low and need to go UP (negative vz)
            alt_error = current_dist - TARGET_ALTITUDE
            vz = float(-alt_error * 1.5) # Soft correction back to 3m
            
            status = "SEARCHING / LEVELLING"
            
            if gate_in_view and (time.time() - last_passed_time > 1.5):
                gate_count += 1
                last_passed_time = time.time()
                print(f"\n>>>>>>> GATE {gate_count} CLEARED <<<<<<<\n")
                gate_in_view = False

        # --- FINAL CRITICAL SAFETY OVERRIDE ---
        if current_dist < MIN_ALTITUDE:
            vz = -3.0 # Hard climb
            status = "!!! CRITICAL LOW !!!"
        elif current_dist > MAX_ALTITUDE:
            vz = 2.0  # Hard descend

        print(f"{status} | Conf: [{bar}] {conf:.2f} | vz: {vz:+.2f}")

        # Execute Racing Move
        client.moveByVelocityAsync(vx, 0, vz, duration=0.1, 
                                 yaw_mode=airsim.YawMode(True, yaw_rate))

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nManual Stop.")

finally:
    client.hoverAsync().join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)