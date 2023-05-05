from pyrplidar import PyRPlidar
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import heapq

#change the starting point here to navigate the robot
starting_point = (0,0)
destination_point = (0,1000)

lidar = PyRPlidar()
lidar.connect(port="/dev/ttyUSB0", baudrate=256000, timeout=3)
scan_generator = lidar.start_scan()

# Define grid parameter
grid_resolution = 100  # mm per cell
grid_size = (30, 30)  # Define your grid size according to your data

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




def plot_data(x_values, y_values, start, dest, path=None):
    plt.scatter(x_values, y_values, s=2, label='LIDAR data')
    plt.scatter(*start, s=50, c='green', marker='o', label='Starting Point')
    plt.scatter(*dest, s=50, c='red', marker='x', label='Destination Point')
    
    if path:
        path_x, path_y = zip(*path)
        plt.scatter(path_x, path_y, s=50, c='blue', marker='.', label='A* Path')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("LIDAR Data Visualization")
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.legend()
    plt.show()



iterations = 0
max_iterations = 100

direction_angle = 90  # Set the desired direction angle here (in degrees)
angle_range = 10  # Set the angle range around the direction angle (in degrees)


def update_occupancy_grid(x_values, y_values):
    for x, y in zip(x_values, y_values):
        i, j = int(x // grid_resolution), int(y // grid_resolution)
        if 0 <= i < grid_size[0] and 0 <= j < grid_size[1]:
            occupancy_grid[i, j] = True


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
                continue  # Skip cells with obstacles

            tentative_g_score = g_score[current] + heuristic_cost_estimate(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_cost_estimate(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found








while iterations < max_iterations:
    # x_values, y_values, longest_distance_in_range = update(direction_angle, angle_range)
    x_values, y_values = update()
    print("X values:", x_values)
    print("Y values:", y_values)
    # print("Longest distance in the range {:.1f}° to {:.1f}°: {:.2f}".format(direction_angle - angle_range / 2, direction_angle + angle_range / 2, longest_distance_in_range))
    
    # plot_data(x_values, y_values, starting_point, destination_point)

    update_occupancy_grid(x_values,y_values)

    iterations += 1

# Convert starting and destination points to grid coordinates
start_grid = (starting_point[0] // grid_resolution, starting_point[1] // grid_resolution)
goal_grid = (destination_point[0] // grid_resolution, destination_point[1] // grid_resolution)

# Find the path using A* algorithm
path = a_star(start_grid, goal_grid, occupancy_grid)
if path:
    print("Path found!")
    path_coordinates = [(cell[0] * grid_resolution, cell[1] * grid_resolution) for cell in path]
    print("Path coordinates:", path_coordinates)
else:
    print("No path found!")

# Plot the occupancy grid
plt.imshow(occupancy_grid.T, origin='lower', cmap='gray_r')
plt.colorbar()
plt.show()


# Plot the LIDAR data and the path found by the A* algorithm
plot_data(x_values, y_values, starting_point, destination_point, path_coordinates if path else None)


lidar.stop()
lidar.disconnect()