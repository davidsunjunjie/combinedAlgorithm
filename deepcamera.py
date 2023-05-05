import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

openni2.initialize('/opt/openni-sdk/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs')  # Replace with the path to your OpenNI SDK library

# Open a device
dev = openni2.Device.open_any()

# Create a depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()

# Define the distance threshold for obstacles (in millimeters)
obstacle_threshold = 1000

while True:
    # Read a depth frame from the camera
    frame = depth_stream.read_frame()
    depth_data = frame.get_buffer_as_uint16()
    depth_image = np.frombuffer(depth_data, dtype=np.uint16)
    depth_image.shape = (frame.height, frame.width)

    # Threshold the depth image to identify obstacles
    obstacle_mask = depth_image < obstacle_threshold

    # Calculate the centroid of the obstacles
    y, x = np.where(obstacle_mask)
    if len(y) > 0:
        centroid = (np.mean(x), np.mean(y))
        print("Obstacle detected at centroid:", centroid)
    else:
        print("No obstacles detected")

# Stop the camera and unload the library
depth_stream.stop()
openni2.unload()