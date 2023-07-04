from utils.cpu_operations import cpu_operations

def main():

    # Create an instance of the cpu_operations class
    cpu_operat = cpu_operations()
    matrix_size = 1000
    num_executions = 10

    # Perform matrix multiplication on CPU
    average_execution_time = cpu_operat.calculate_average_execution_time(matrix_size,num_executions)
    print("Average Execution Time:", average_execution_time, "seconds")

if __name__ == '__main__':
    main()
