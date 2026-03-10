import airsim
client = airsim.MultirotorClient()
# Get the actual ID that AirSim is currently using for that object
actual_id = client.simGetSegmentationObjectID("Gate")
print(f"The actual ID for 'Gate' is: {actual_id}")