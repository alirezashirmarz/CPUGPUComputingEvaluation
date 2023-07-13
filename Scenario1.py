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

import numpy as np
import cupy as cp
import tensorflow as tf
import torch
import time

# Define the matrix sizes
matrix_size = 1000
test_count = 100
matrix_pairs = []

for i in range(test_count):
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)
    matrix_pairs.append([matrix_a, matrix_b])


# CPU Matrix Multiplication
def cpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using CPU
    cpu_result = np.dot(matrix_a, matrix_b)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time


# GPU Matrix Multiplication
def gpu_matrix_multiplication_cupy(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using GPU
    gpu_matrix_a = cp.asarray(matrix_a)
    gpu_matrix_b = cp.asarray(matrix_b)
    gpu_result = cp.dot(gpu_matrix_a, gpu_matrix_b)
    cpu_result = cp.asnumpy(gpu_result)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time


def gpu_matrix_multiplication_tensorflow(matrix_a, matrix_b):
    start_time = time.time()

    with tf.device('/GPU:0'):
      gpu_result = tf.matmul(matrix_a, matrix_b)

    end_time = time.time()
    execution_time = end_time - start_time
    return gpu_result, execution_time


def gpu_matrix_multiplication_pytorch(matrix_a, matrix_b):
    start_time = time.time()

    gpu_matrix_a = torch.from_numpy(matrix_a)
    gpu_matrix_b = torch.from_numpy(matrix_b)
    gpu_result = torch.matmul(gpu_matrix_a, gpu_matrix_b)
    gpu_result = gpu_result.numpy()

    end_time = time.time()
    execution_time = end_time - start_time
    return gpu_result, execution_time


def cpu_test():
    test_result = []

    for i in range(test_count):
        result, execution_time = cpu_matrix_multiplication(matrix_a, matrix_b)
        # test_result.append([result, execution_time])
        test_result.append(execution_time)

    return test_result


def gpu_test(mul_function):
    test_result = []

    for i in range(test_count):
        result, execution_time = mul_function(matrix_a, matrix_b)
        # test_result.append([result, execution_time])
        test_result.append(execution_time)

    return test_result


# Run CPU Matrix Multiplication
cpu_test_result = cpu_test()

# Run GPU Matrix Multiplication
gpu_test_result_cupy = gpu_test(gpu_matrix_multiplication_cupy)
gpu_test_result_tensorflow = gpu_test(gpu_matrix_multiplication_tensorflow)
gpu_test_result_pytorch = gpu_test(gpu_matrix_multiplication_pytorch)

# Compare the results (optional)
# print("CPU Result:")
# print(cpu_result)
# print("GPU Result:")
# print(gpu_result)

# Print the execution times
print("CPU Execution Time:              ", cpu_test_result, "seconds")
print("GPU Execution Time (CuPy):       ", gpu_test_result_cupy, "seconds")
print("GPU Execution Time (TensorFlow): ", gpu_test_result_tensorflow, "seconds")
print("GPU Execution Time (PyTorch):    ", gpu_test_result_pytorch, "seconds")

print("\n")

print("CPU Execution Time Average:              ", sum(cpu_test_result) / test_count, "seconds")
print("GPU Execution Time Average (CuPy):       ", sum(gpu_test_result_cupy) / test_count, "seconds")
print("GPU Execution Time Average (TensorFlow): ", sum(gpu_test_result_tensorflow) / test_count, "seconds")
print("GPU Execution Time Average (PyTorch):    ", sum(gpu_test_result_pytorch) / test_count, "seconds")

