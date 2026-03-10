import airsim
client = airsim.MultirotorClient()
client.confirmConnection()

# This forces AirSim to assign ID 100 to anything with "Gate" in the name
success = client.simSetSegmentationObjectID("Gate.*", 100, True)
print(f"Gate ID Assignment Successful: {success}")