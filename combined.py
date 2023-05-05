import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api
from pyrplidar import PyRPlidar
from queue import PriorityQueue
import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches



#initialize the deep sensing camera

openni2.initialize('/opt/openni-sdk/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs')  # Replace with the path to your OpenNI SDK library

# Open a device
dev = openni2.Device.open_any()

# Create a depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()

# Create a color stream
color_stream = dev.create_color_stream()
color_stream.start()



#initialize the compatible LIDAR sensor 

lidar = PyRPlidar()
lidar.connect(port="/dev/ttyUSB0", baudrate=256000, timeout=3)
scan_generator = lidar.start_scan()



# Define grid parameters
grid_resolution = 100  # mm per cell
grid_size = (30, 30)  # Define your grid size according to your data

# Define the starting_point and destination_point 
starting_point = (0, 0)
destination_point = (0, 1500)

distance_threshold = 10000  # Set the distance threshold here (in millimeters)
max_iterations = 100

# Create the occupancy grid
occupancy_grid = np.zeros(grid_size, dtype=bool)

def update():
    angles, distances = [], []
    distance_threshold = 10000  # Set the distance threshold here (in millimeters)

    for _ in range(500):  # You can adjust this value according to your needs
        try:
            data = next(scan_generator())
        except StopIteration:
            print("LIDAR scan has finished.")
            break

        if data.distance > distance_threshold:  # Ignore data points beyond the distance threshold
            continue

        angle_rad = np.radians(data.angle)
        x = data.distance * np.cos(angle_rad)
        y = data.distance * np.sin(angle_rad)

        angles.append(x)
        distances.append(y)

    return angles, distances


def update_depth_data():
    # Read a depth frame
    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    
    # Preprocess the data
    depth_image = np.frombuffer(depth_data, dtype=np.uint16)
    depth_image.shape = (depth_frame.height, depth_frame.width)

    return depth_image

def update_combined_occupancy_grid(depth_image, angles, distances):
    # Update the occupancy grid using both depth camera and LIDAR data
    # Give higher priority to depth camera data

    # Update occupancy grid using depth camera data
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            depth = depth_image[i, j]
            if depth > distance_threshold:  # Ignore data points beyond the distance threshold
                continue
            
            x = depth * (j - depth_image.shape[1] // 2) / depth_image.shape[1]
            y = depth * (i - depth_image.shape[0] // 2) / depth_image.shape[0]

            grid_i, grid_j = int(x // grid_resolution), int(y // grid_resolution)
            if 0 <= grid_i < grid_size[0] and 0 <= grid_j < grid_size[1]:
                occupancy_grid[grid_i, grid_j] = True

    # Update occupancy grid using LIDAR data
    for angle, distance in zip(angles, distances):
        if distance > distance_threshold:  # Ignore data points beyond the distance threshold
            continue

        angle_rad = np.radians(angle)
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)

        grid_i, grid_j = int(x // grid_resolution), int(y // grid_resolution)
        if 0 <= grid_i < grid_size[0] and 0 <= grid_j < grid_size[1]:
            occupancy_grid[grid_i, grid_j] = True



def visualize_occupancy_grid(occupancy_grid):
    fig, ax = plt.subplots()
    
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i, j]:
                cell = patches.Rectangle((i * grid_resolution, j * grid_resolution), grid_resolution, grid_resolution, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(cell)
            else:
                cell = patches.Rectangle((i * grid_resolution, j * grid_resolution), grid_resolution, grid_resolution, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(cell)
    
    ax.set_xlim(0, grid_size[0] * grid_resolution)
    ax.set_ylim(0, grid_size[1] * grid_resolution)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def depth_to_xy(depth_image):
    x_values, y_values = [], []

    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            depth = depth_image[i, j]
            if depth > distance_threshold:  # Ignore data points beyond the distance threshold
                continue

            x = depth * (j - depth_image.shape[1] // 2) / depth_image.shape[1]
            y = depth * (i - depth_image.shape[0] // 2) / depth_image.shape[0]

            x_values.append(x)
            y_values.append(y)

    return x_values, y_values





def plot_data(x_values, y_values, depth_x_values, depth_y_values, start, dest, path=None):
    plt.scatter(x_values, y_values, s=2, label='LIDAR data')
    plt.scatter(depth_x_values, depth_y_values, s=2, c='cyan', label='Depth camera data')
    plt.scatter(*start, s=50, c='green', marker='o', label='Starting Point')
    plt.scatter(*dest, s=50, c='red', marker='x', label='Destination Point')

    if path:
        path_x, path_y = zip(*path)
        plt.scatter(path_x, path_y, s=50, c='blue', marker='.', label='A* Path')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("LIDAR and Depth Camera Data Visualization")
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.legend()
    plt.show()





def main():
    while True:
        # Update depth data and LIDAR data
        depth_image = update_depth_data()
        angles, distances = update()  # Directly unpack the two lists returned by update()

        depth_x_values, depth_y_values = depth_to_xy(depth_image)

        # Update the occupancy grid
        update_combined_occupancy_grid(depth_image, angles, distances)

        # Visualize the updated occupancy grid
        visualize_occupancy_grid(occupancy_grid)

        # Plot the LIDAR data, depth camera data, starting point, destination point, and A* path (if any)
        plot_data(angles, distances, depth_x_values, depth_y_values, starting_point, destination_point)  # You can add the path variable if you have one

        # Add a delay if needed
        time.sleep(0.1)




# Call the main function
if __name__ == "__main__":
    main()
