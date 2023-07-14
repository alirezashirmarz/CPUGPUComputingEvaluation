import json
import matplotlib.pyplot as plt

# Read the results from the JSON file
with open("results.json", "r") as f:
    results = json.load(f)

# Extract the matrix sizes, CPU times, and GPU times from the results
matrix_sizes = [result["matrix_size"] for result in results]
cpu_times = [result["cpu_time"] for result in results]
gpu_times = [result["gpu_time"] for result in results]

# Plot the CPU and GPU times as line plots
plt.plot(matrix_sizes, cpu_times, label="CPU")
plt.plot(matrix_sizes, gpu_times, label="GPU")

# Set the plot title and axis labels
plt.title("Performance Comparison of CPU and GPU for Matrix Multiplication")
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (seconds)")

# Add a legend to the plot
plt.legend()

# Display the plot
plt.show()