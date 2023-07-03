import numpy as np
import time

class CPUMatrixOperations:
    def __init__(self):
        pass

    def matrix_multiplication(self, matrix_size=1000):
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        start_time = time.time()

        # Perform matrix multiplication using CPU
        cpu_result = np.dot(matrix_a, matrix_b)

        end_time = time.time()
        execution_time = end_time - start_time
        return cpu_result, execution_time
