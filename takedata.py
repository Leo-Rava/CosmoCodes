import time
import csv
import os
from labjack import ljm

AIN_CHANNEL = "AIN0"
SAMPLE_RATE_HZ = 100
DURATION_SEC = 10
OUTPUT_FILENAME = "labjack_data.csv"


dt = 1.0 / SAMPLE_RATE_HZ
num_samples = int(SAMPLE_RATE_HZ * DURATION_SEC)

handle = ljm.openS("ANY", "ANY", "ANY")

info = ljm.getHandleInfo(handle)
print(f"Opened LabJack: DeviceType={info[0]}, Serial={info[2]}, Connection={info[1]}")

ljm.eWriteName(handle, f"{AIN_CHANNEL}_RANGE", 10.0)
ljm.eWriteName(handle, f"{AIN_CHANNEL}_RESOLUTION_INDEX", 1)

with open(OUTPUT_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp (s)", f"{AIN_CHANNEL} (V)"])

    print(f"\nRecording {DURATION_SEC}s of data at {SAMPLE_RATE_HZ} Hz...")
    print(f"Writing to '{OUTPUT_FILENAME}'")

    start_time = time.time()

    for i in range(num_samples):
        loop_start = time.time()

        voltage = ljm.eReadName(handle, AIN_CHANNEL)

        timestamp = loop_start - start_time

        writer.writerow([timestamp, voltage])
        elapsed = time.time() - loop_start
        sleep_time = max(0, dt - elapsed)
        time.sleep(sleep_time)

    print("\nData collection complete.")

ljm.close(handle)
print("LabJack closed.")
