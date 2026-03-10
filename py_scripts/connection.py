import airsim

def init_client():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    print("[AirSim] Connected successfully.")
    return client