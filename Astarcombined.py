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
import serial


# sending the command to the robot to carry out
# def send_instruction(instruction):
#     try:
#         ser = serial.Serial("/dev/ttyUSB1", 9600)  # Replace YOUR_SERIAL_PORT with the appropriate device name
#         ser.write(instruction.encode())
#         ser.close()
#     except Exception as e:
#         print("Error sending instruction:", e)
  
 


# def connect_serial(port, baudrate=9600):
#     ser = serial.Serial(port, baudrate)
#     return ser



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
grid_size = (40, 40)  # Define your grid size according to your data

#distance threhold
distance_threshold =  1000

# Define the starting_point and destination_point 
starting_point = (0, 0)
destination_point = (0, 800)



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

        angles.append(data.angle)
        distances.append(data.distance)

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

    min_distance_threshold = 10  # Set the minimum distance threshold here (in millimeters)

    # Update occupancy grid using depth camera data
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            depth = depth_image[i, j]
            if not min_distance_threshold < depth < distance_threshold:  # Ignore data points beyond the distance threshold and too close to the sensor
                continue

            x = depth * (j - depth_image.shape[1] // 2) / depth_image.shape[1]
            y = depth * (i - depth_image.shape[0] // 2) / depth_image.shape[0]

            grid_i, grid_j = int(x // grid_resolution), int(y // grid_resolution)
            if 0 <= grid_i < grid_size[0] and 0 <= grid_j < grid_size[1]:
                occupancy_grid[grid_i, grid_j] = True

    # Update occupancy grid using LIDAR data
    for angle, distance in zip(angles, distances):
        if not min_distance_threshold < distance < distance_threshold:  # Ignore data points beyond the distance threshold and too close to the sensor
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
    plt.scatter(y_values, x_values, s=2, label='LIDAR data')  # Swap x and y values
    plt.scatter(depth_y_values, depth_x_values, s=2, c='cyan', label='Depth camera data')  # Swap x and y values
    plt.scatter(*start, s=50, c='green', marker='o', label='Starting Point')
    plt.scatter(*dest, s=50, c='red', marker='x', label='Destination Point')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, c='blue', marker='.', label='A* Path', linestyle='-', linewidth=1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("LIDAR and Depth Camera Data Visualization")
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.legend()
    plt.show()


def heuristic_cost_estimate(current, goal):
    return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)


# Define the function to get the neighbors of a cell
def get_neighbors(cell):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            x, y = cell[0] + i, cell[1] + j
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                neighbors.append((x, y))
    return neighbors

def a_star(start, goal, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = dict()
    g_score = {(x, y): float('inf') for x in range(grid_size[0]) for y in range(grid_size[1])}
    g_score[start] = 0
    f_score = {(x, y): float('inf') for x in range(grid_size[0]) for y in range(grid_size[1])}
    f_score[start] = heuristic_cost_estimate(start, goal)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        neighbors = get_neighbors(current)
        for neighbor in neighbors:
            if grid[neighbor[0]][neighbor[1]]:
                continue  # Skip cells with           obstacles

            tentative_g_score = g_score[current] + heuristic_cost_estimate(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_cost_estimate(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


def angles_distances_to_xy(angles, distances):
    x_values, y_values = [], []
    for angle, distance in zip(angles, distances):
        angle_rad = np.radians(angle)
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values

def main():
    iterations = 0
    max_iterations = 3

    while iterations < max_iterations:
        # Update depth data and LIDAR data
        depth_image = update_depth_data()
        angles, distances = update()

        # Update the occupancy grid
        update_combined_occupancy_grid(depth_image, angles, distances)

        iterations += 1

    # Convert angles and distances to x and y values for plotting
    x_values, y_values = angles_distances_to_xy(angles, distances)

    # Get depth_x_values and depth_y_values from depth_image
    depth_x_values, depth_y_values = depth_to_xy(depth_image)

    # Convert starting and destination points to grid coordinates
    start_grid = (starting_point[0] // grid_resolution, starting_point[1] // grid_resolution)
    goal_grid = (destination_point[0] // grid_resolution, destination_point[1] // grid_resolution)

    # Calculate A* path
    path = a_star(start_grid, goal_grid, occupancy_grid)

    # arduino_port = "/dev/ttyUSB1"
    # ser = connect_serial(arduino_port)

    if path:
        print("Path found!")
        path_coordinates = [(cell[0] * grid_resolution, cell[1] * grid_resolution) for cell in path]
        print("Path coordinates:", path_coordinates)
        
    #     for i, coord in enumerate(path_coordinates):
    #         if i < len(path_coordinates) - 1:
    #             next_coord = path_coordinates[i + 1]
    #             dx = next_coord[0] - coord[0]
    #             dy = next_coord[1] - coord[1]
                
    #             if dx > 0 and dy == 0:
    #                 send_instruction("right")
    #             elif dx < 0 and dy == 0:
    #                 send_instruction("left")
    #             elif dx == 0 and dy > 0:
    #                 send_instruction("up")
    #             elif dx == 0 and dy < 0:
    #                 send_instruction("down")
    #             elif dx > 0 and dy > 0:
    #                 send_instruction("up_right")
    #             elif dx > 0 and dy < 0:
    #                 send_instruction("down_right")
    #             elif dx < 0 and dy > 0:
    #                 send_instruction("up_left")
    #             elif dx < 0 and dy < 0:
    #                 send_instruction("down_left")
                
    #             # Add a delay between instructions if necessary
    #             time.sleep(0.1)

    else:
        print("No path found!")
    
    # ser.close()

    # Plot the occupancy grid
    plt.imshow(occupancy_grid.T, origin='lower', cmap='gray_r')
    plt.colorbar()
    plt.show()

    plot_data(x_values, y_values, depth_x_values, depth_y_values, starting_point, destination_point, path_coordinates if path else None)

# Call the main function
if __name__ == "__main__":
    main()


