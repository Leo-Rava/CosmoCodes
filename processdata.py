import numpy as np
import matplotlib.pyplot as plt
import os

input_csv = "labjack_data.csv"       # Input file
output_csv = "filtered_data.csv"     # Output file
filter_type = "exponential"          # "moving_average" or "exponential"
integration_time = 2.0               # Time constant (seconds)


df = np.loadtxt(input_csv, delimiter=",")

time = df[:, 0]
signal = df[:, 1]

# Estimate sample interval
if len(time) < 2:
    raise ValueError("Not enough data to estimate sampling rate.")
dt = np.median(np.diff(time))  # Assume mostly regular sampling

# --- Apply Filtering ---
filtered = np.zeros_like(signal)

if filter_type == "moving_average":
    window_size = int(integration_time / dt)
    if window_size < 1:
        raise ValueError("Integration time too small for sampling rate.")
    print(f"Using moving average with window size: {window_size} samples")
    for i in range(len(signal)):
        start = max(0, i - window_size + 1)
        filtered[i] = np.mean(signal[start:i+1])

elif filter_type == "exponential":
    alpha = dt / integration_time
    print(f"Using exponential filter with alpha = {alpha:.6f}")
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = (1 - alpha) * filtered[i - 1] + alpha * signal[i]

else:
    raise ValueError(f"Unknown filter type: {filter_type}")



plt.figure(figsize=(10, 5))
plt.plot(time, signal, label="Raw", alpha=0.5)
plt.plot(time, filtered, label="Filtered", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Signal Filtering")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()