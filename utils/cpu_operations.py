import json
import numpy as np
import time
from .CPUMatrixOperations import matrix_multiplication

class cpu_operations:
    

    # save_to_json 
    def save_to_json(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def calculate_average_execution_time(self, matrix_size, num_executions):
        total_execution_time = 0

        for _ in range(num_executions):
            _, cpu_execution_time =matrix_multiplication(matrix_size)
            total_execution_time += cpu_execution_time

            average_execution_time = total_execution_time / num_executions
        return average_execution_time

    
    
