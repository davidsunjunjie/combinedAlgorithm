from pyrplidar import PyRPlidar
import matplotlib.pyplot as plt
import numpy as np



lidar = PyRPlidar()
lidar.connect(port="/dev/ttyUSB0", baudrate=256000, timeout=3)
lidar.reset()






lidar.stop()
lidar.disconnect()