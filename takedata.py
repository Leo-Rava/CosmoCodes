import time
import csv
import os
from labjack import ljm

AIN_CHANNEL = "AIN0"
SAMPLE_RATE_HZ = 100
DURATION_SEC = 10
OUTPUT_FILENAME =

# Compute sample interval and number of samples
dt = 1.0 / SAMPLE_RATE_HZ
num_samples = int(SAMPLE_RATE_HZ * DURATION_SEC)

# Open connection to LabJack (any type, connection, device)
handle = ljm.openS("ANY", "ANY", "ANY")

# Device Info
info = ljm.getHandleInfo(handle)
print(f"Opened LabJack: DeviceType={info[0]}, Serial={info[2]}, Connection={info[1]}")

# Optional: configure analog input range and resolution
ljm.eWriteName(handle, f"{AIN_CHANNEL}_RANGE", 10.0)  # ±10 V range
ljm.eWriteName(handle, f"{AIN_CHANNEL}_RESOLUTION_INDEX", 1)

# Open CSV file for writing
with open(OUTPUT_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp (s)", f"{AIN_CHANNEL} (V)"])  # Header

    print(f"\nRecording {DURATION_SEC}s of data at {SAMPLE_RATE_HZ} Hz...")
    print(f"Writing to '{OUTPUT_FILENAME}'")

    start_time = time.time()

    for i in range(num_samples):
        loop_start = time.time()

        # Read analog input
        voltage = ljm.eReadName(handle, AIN_CHANNEL)

        # Timestamp relative to start
        timestamp = loop_start - start_time

        # Write to CSV
        writer.writerow([timestamp, voltage])

        # Wait for next sample
        elapsed = time.time() - loop_start
        sleep_time = max(0, dt - elapsed)
        time.sleep(sleep_time)

    print("\nData collection complete.")

# Close device
ljm.close(handle)
print("LabJack closed.")
