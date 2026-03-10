import airsim
import cv2
import numpy as np
import os
import random
import time

# --- CONFIG ---
SAVE_PATH = "SNN_Gate_Dataset/train"
NUM_SAMPLES = 1000
X_Y_RANGE = [-7, 7]
Z_SAFE_ZONE = [-4.0, -6.0]

os.makedirs(f"{SAVE_PATH}/images", exist_ok=True)
os.makedirs(f"{SAVE_PATH}/labels", exist_ok=True)

client = airsim.MultirotorClient()
client.confirmConnection()

def get_best_target_id(seg_img):
    """
    Scans the frame and returns the ID of the object that 
    most likely looks like a gate (Square-ish and medium sized).
    """
    unique_ids = np.unique(seg_img.reshape(-1, 3), axis=0)
    for uid in unique_ids:
        if np.all(uid == 0): continue # Skip sky
        
        mask = np.all(seg_img == uid, axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = float(w) / h
            
            # --- THE GATE HEURISTIC ---
            # Is it square-ish (0.5-2.0) and not the whole floor (>500px)?
            if 0.5 < aspect_ratio < 2.0 and 400 < area < (seg_img.shape[0] * seg_img.shape[1] * 0.3):
                return uid[0] # Found a likely gate ID!
    return None

print("Starting Auto-Discovery Collector...")
count = 0
last_known_id = 231 # Start with your last successful ID

while count < NUM_SAMPLES:
    try:
        # Teleport
        rx, ry, rz = random.uniform(*X_Y_RANGE), random.uniform(*X_Y_RANGE), random.uniform(Z_SAFE_ZONE[0], Z_SAFE_ZONE[1])
        yaw = np.arctan2(-ry, -rx)
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(rx, ry, rz), airsim.to_quaternion(0, 0, yaw)), True)
        
        time.sleep(0.6)
        
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
        ])

        if not responses or len(responses) < 2: continue

        img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
        img_seg = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height, responses[1].width, 3)
        
        # 1. Try to find the gate ID dynamically
        current_id = get_best_target_id(img_seg)
        
        # 2. If we found a gate, update our "Last Known" and save
        if current_id is not None:
            if current_id != last_known_id:
                print(f"ID Shift Detected! Now locked onto ID: {current_id}")
                last_known_id = current_id
            
            # Draw for the Live View
            debug_view = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Re-mask with the confirmed ID to get coordinates
            mask = (img_seg[:,:,0] == last_known_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            labels = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if cv2.contourArea(cnt) < 200: continue
                
                # Draw high-contrast Blue Box
                cv2.rectangle(debug_view, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # YOLO Normalize
                cx, cy = (x + w/2.0) / responses[0].width, (y + h/2.0) / responses[0].height
                nw, nh = w / responses[0].width, h / responses[0].height
                labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if labels:
                fn = f"gate_{count:04d}"
                cv2.imwrite(os.path.join(SAVE_PATH, "images", f"{fn}.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                with open(os.path.join(SAVE_PATH, "labels", f"{fn}.txt"), "w") as f:
                    f.write("\n".join(labels))
                count += 1
                if count % 10 == 0: print(f"Captured {count}/1000")
            
            cv2.imshow("Auto-Target View", debug_view)
        else:
            print("Searching for gate ID...")
            cv2.imshow("Auto-Target View", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(1)

cv2.destroyAllWindows()