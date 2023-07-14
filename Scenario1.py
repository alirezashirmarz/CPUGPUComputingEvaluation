# ************************ Written by Alireza *********modified by Kiani*************************
import numpy as np
import time
import cupy as cp
import numba
import json

# Define the matrix sizes
matrix_sizes = [200, 1000, 2000, 3000, 4000, 5000]

# Define a list to store the results
results = []

for matrix_size in matrix_sizes:
    print(f"\nMatrix size: {matrix_size}")
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    # CPU Matrix Multiplication
    cpu_total_time = 0
    iterations = 5
    for i in range(iterations):
        start_time = time.time()
        cpu_result = np.dot(matrix_a, matrix_b)
        end_time = time.time()
        cpu_total_time += end_time - start_time
    cpu_avg_time = cpu_total_time / iterations

    # GPU Matrix Multiplication
    gpu_total_time = 0
    iterations = 5
    for i in range(iterations):
        start_time = time.time()
        gpu_matrix_a = cp.asarray(matrix_a)
        gpu_matrix_b = cp.asarray(matrix_b)

        @numba.jit(forceobj=True, parallel=True)
        def g_matrix_multiplication(a, b):
            return a @ b

        gpu_result = g_matrix_multiplication(gpu_matrix_a, gpu_matrix_b)
        end_time = time.time()
        gpu_total_time += end_time - start_time
    gpu_avg_time = gpu_total_time / iterations

    # Compare the average execution times
    print("Average CPU Execution Time:", cpu_avg_time, "seconds")
    print("Average GPU Execution Time:", gpu_avg_time, "seconds")

    # Store the results in a dictionary
    result = {"matrix_size": matrix_size, "cpu_time": cpu_avg_time, "gpu_time": gpu_avg_time}
    results.append(result)

# Write the results to a JSON file
with open("results.json", "w") as f:
    json.dump(results, f)

""" The Explanation of the Code:
In this code, we start by defining the size of the square matrices to be multiplied (matrix_size). 
We then generate two random matrices (matrix_a and matrix_b) using the np.random.rand function.

We define two functions: cpu_matrix_multiplication and gpu_matrix_multiplication, 
which perform matrix multiplication using CPU and GPU, respectively. 

In the CPU function, we use the NumPy np.dot function for matrix multiplication. 
In the GPU function, we use the cp.asarray function from the cupy library 
to transfer the matrices to the GPU, perform matrix multiplication using cp.dot, 
and then transfer the result back to the CPU using cp.asnumpy.

After running the CPU and GPU matrix multiplication functions, 
we can optionally compare the results and print the execution times.

Please note that for the GPU matrix multiplication to work, 
you need to have the cupy library installed, which provides 
a NumPy-compatible interface for GPU computations. 
Also, make sure you have a compatible GPU and CUDA drivers installed.
for the gpu accelerated library we use Numba which can optimize gpu performance
you can change elements of matrix size array to compare average times of cpu and gpu also
you can change number of iteration for each function

The code also saves the average cpu and gpu time and matrix sizes to a json file
"""
