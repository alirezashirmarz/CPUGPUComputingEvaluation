from utils.CPUMatrixOperations import CPUMatrixOperations

def main():
    # Create an instance of the CPUMatrixOperations class
    cpu_operations = CPUMatrixOperations()

    # Perform matrix multiplication on CPU
    cpu_result, cpu_execution_time = cpu_operations.matrix_multiplication(1000)

    # Compare the results (optional)
    print("Result in CPU:")
    print(cpu_result)

    # Display the execution time
    print("Execution time in CPU:", cpu_execution_time, "seconds")

if __name__ == '__main__':
    main()
