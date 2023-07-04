import json
from utils.cpu_operations import cpu_operations

def main():

    # Create an instance of the cpu_operations class
    cpu_operat = cpu_operations()
    matrix_size = 1000
    num_executions = 10

    # Perform matrix multiplication on CPU
    cpu_operat.calculate_average_execution_time(matrix_size, num_executions)

    cpu_operat.calculate_and_save_averaged_results()


if __name__ == '__main__':
    main()


