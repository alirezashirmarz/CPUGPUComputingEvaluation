from utils import *

# Run CPU Matrix Multiplication
cpu_result, cpu_execution_time = CPUMatrixOperations.cpu_matrix_multiplication(1000)


# Compare the results (optional)
print("CPU Result:")
print(cpu_result)


# Print the execution times
print("CPU Execution Time:", cpu_execution_time, "seconds")

