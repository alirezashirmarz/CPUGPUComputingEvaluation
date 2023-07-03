# ************************ Written by Alireza ************************************
import numpy as np
import time

# Define the matrix sizes
matrix_size = 1000
matrix_a = np.random.rand(matrix_size, matrix_size)
matrix_b = np.random.rand(matrix_size, matrix_size)

# CPU Matrix Multiplication
def cpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using CPU
    cpu_result = np.dot(matrix_a, matrix_b)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

# GPU Matrix Multiplication
def gpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using GPU
    gpu_matrix_a = cp.asarray(matrix_a)
    gpu_matrix_b = cp.asarray(matrix_b)
    gpu_result = cp.dot(gpu_matrix_a, gpu_matrix_b)
    cpu_result = cp.asnumpy(gpu_result)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

# Run CPU Matrix Multiplication
cpu_result, cpu_execution_time = cpu_matrix_multiplication(matrix_a, matrix_b)

# Run GPU Matrix Multiplication
gpu_result, gpu_execution_time = gpu_matrix_multiplication(matrix_a, matrix_b)

# Compare the results (optional)
print("CPU Result:")
print(cpu_result)
print("GPU Result:")
print(gpu_result)

# Print the execution times
print("CPU Execution Time:", cpu_execution_time, "seconds")
print("GPU Execution Time:", gpu_execution_time, "seconds")

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

Feel free to adjust the matrix size and modify the code according to your specific requirements.
"""
