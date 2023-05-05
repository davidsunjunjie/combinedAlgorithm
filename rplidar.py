from pyrplidar import PyRPlidar
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation




lidar = PyRPlidar()
lidar.connect(port="/dev/ttyUSB0", baudrate=256000, timeout=3)




scan_generator = lidar.start_scan()

count = 0
while(True):
    for scan_data in scan_generator():
        print(scan_data)
    count+=1
    if count == 1000: 
        break


lidar.stop()
lidar.disconnect()