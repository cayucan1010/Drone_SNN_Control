import airsim
import cv2
import numpy as np
import os
import random
import time

# --- CONFIG ---
SAVE_PATH = "SNN_Gate_Dataset/train"
GATE_ID = 225  # Updated based on your channel_scraper results
NUM_SAMPLES = 100
X_RANGE, Y_RANGE, Z_RANGE = [-7, 7], [-7, 7], [-2, -6]

os.makedirs(f"{SAVE_PATH}/images", exist_ok=True)
os.makedirs(f"{SAVE_PATH}/labels", exist_ok=True)

client = airsim.MultirotorClient()
client.confirmConnection()

def get_yolo_labels(seg_img, img_w, img_h):
    # Based on your scraper, 225 was found in the Blue channel (index 0)
    gate_mask = (seg_img[:,:,0] == GATE_ID).astype(np.uint8)
    
    contours, _ = cv2.findContours(gate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    labels = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 120: continue # Filters out distant tiny specks
        x, y, w, h = cv2.boundingRect(cnt)
        
        # YOLO Normalization
        cx = (x + w/2.0) / img_w
        cy = (y + h/2.0) / img_h
        nw = w / img_w
        nh = h / img_h
        labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return labels

print(f"Starting collection for ID {GATE_ID}. Folders should begin filling now...")

for i in range(NUM_SAMPLES):
    # 1. Teleport
    rand_x = random.uniform(*X_RANGE)
    rand_y = random.uniform(*Y_RANGE)
    rand_z = random.uniform(*Z_RANGE)
    
    # Force look-at origin (assuming gate is near 0,0)
    angle_to_center = np.arctan2(-rand_y, -rand_x)
    
    pose = airsim.Pose(airsim.Vector3r(rand_x, rand_y, rand_z), 
                       airsim.to_quaternion(0, 0, angle_to_center))
    
    client.simSetVehiclePose(pose, ignore_collision=True)
    time.sleep(0.4) # Give the GPU a moment to draw the stencil

    # 2. Capture
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
    ])

    if len(responses) < 2: continue

    img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    img_seg = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height, responses[1].width, 3)

    # 3. Label
    yolo_labels = get_yolo_labels(img_seg, responses[0].width, responses[0].height)
    
    if yolo_labels:
        img_name = f"gate_{i:04d}.png"
        txt_name = f"gate_{i:04d}.txt"
        
        cv2.imwrite(os.path.join(SAVE_PATH, "images", img_name), img_rgb)
        with open(os.path.join(SAVE_PATH, "labels", txt_name), "w") as f:
            f.write("\n".join(yolo_labels))
        
        if i % 10 == 0:
            print(f"Progress: {i}/{NUM_SAMPLES} samples saved.")
    else:
        if i % 25 == 0:
            print(f"Sample {i}: Gate not in view (ID {GATE_ID} missing from blue channel)")

print(f"Done! Check {SAVE_PATH}/images for your data.")