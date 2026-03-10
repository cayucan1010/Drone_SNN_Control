import collections.abc
import sys
import time
import os
import airsim
import numpy as np


# --- STEP 1: PYTHON 3.14 LEGACY PATCHES ---
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

try:
    import airsim
except ImportError:
    print("Error: airsim module not found. Make sure your (.venv) is active!")
    sys.exit(1)

# Connect to the AirSim simulator
client = airsim.MultirotorClient(ip="127.0.0.1")

def run_movement_test():
    try:
        print("--- Connecting to AirSim ---")
        client.confirmConnection()
        
        print("Enabling API control and arming...")
        client.enableApiControl(True)
        client.armDisarm(True)

        # 1. Takeoff
        print("Taking off...")
        client.takeoffAsync().join()
        
        # 2. Move Up to a safe height (5 meters up)
        print("Gaining altitude...")
        client.moveToZAsync(-5, 2).join()

        # 3. Move Forward (X = 5m/s, Y = 0, Z = 0) for 3 seconds
        print("Moving FORWARD at 5m/s...")
        client.moveByVelocityAsync(5, 0, 0, 3).join()
        
        # 4. Hover for a second
        client.hoverAsync().join()
        time.sleep(1)

        # 5. Move Right (X = 0, Y = 5m/s, Z = 0) for 3 seconds
        print("Moving RIGHT at 5m/s...")
        client.moveByVelocityAsync(0, 5, 0, 3).join()

        # 6. Move Backward and Down (X = -5m/s, Y = 0, Z = 2m/s)
        # This creates a diagonal descent
        print("Moving BACKWARD and DOWN...")
        client.moveByVelocityAsync(-5, 0, 2, 3).join()

        # 7. Final Hover and Land
        print("Landing...")
        client.landAsync().join()
        
        print("Test Complete! Disarming...")
        client.armDisarm(False)
        client.enableApiControl(False)

    except Exception as e:
        print(f"\n[!] AN ERROR OCCURRED: {e}")

if __name__ == "__main__":
    run_movement_test()