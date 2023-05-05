import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a simple function to plot
def func(x, k):
    return np.sin(k * x)

# The function that will be called at each frame to update the plot
def update(frame_number):
    y = func(x, frame_number / 10)
    line.set_ydata(y)
    return line,

# Set up the plot
x = np.linspace(0, 2 * np.pi, 100)
y = func(x, 0)

fig, ax = plt.subplots()
line, = ax.plot(x, y)
ax.set_ylim(-1, 1)

# Create the animation using FuncAnimation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Show the animation
plt.show()
