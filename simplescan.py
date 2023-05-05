import functions.rplidar as rplidar
import time

serial_port = "/dev/ttyUSB0"  # For Linux, e.g., "/dev/ttyUSB0" or "/dev/ttyACM0"
# For Windows, e.g., "COM3" or "COM4"
baud_rate = 115200  # Default baud rate for RPLidar devices

lidar = rplidar.RPLidar(serial_port, baud_rate)

# Start force_scan mode
scan_generator = lidar.force_scan()

try:
    print('Press Ctrl+C to stop...')
    while True:
        # Collect and process the scan data
        for scan_data in scan_generator:
            print(scan_data)
            time.sleep(0.1)
except KeyboardInterrupt:
    print('Stopping...')

# Stop scanning and disconnect from the RPLidar device
lidar.stop()
lidar.disconnect()
