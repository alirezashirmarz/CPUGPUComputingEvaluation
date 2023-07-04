import json
import numpy as np
import time
from .CPUMatrixOperations import matrix_multiplication

class cpu_operations:
    def __init__(self):
        pass
     
        self.matrix_sizes = [100, 500, 1000, 2000]
        self.num_executions = 10

    def save_to_json(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def calculate_average_execution_time(self, matrix_size, num_executions):
        total_execution_time = 0

        for _ in range(num_executions):
            _, cpu_execution_time =matrix_multiplication(matrix_size)
            total_execution_time += cpu_execution_time

            average_execution_time = total_execution_time / num_executions
        output = {
        "matrix_size": matrix_size,
        "average_execution_time": average_execution_time
    }
        
        json_filename = "avg_execution_times_oneMatrix.json"
        self.save_to_json(output, json_filename)
        print(f"avg_execution_times_oneMatrix saved to {json_filename}")
    
    
    def calculate_and_save_averaged_results(self):

        results = {}

        for size in self.matrix_sizes:
            average_execution_time = self.calculate_average_execution_time(size, self.num_executions)
            results[size] = average_execution_time

        averaged_results = {}

        for size in self.matrix_sizes:
            averaged_results[size] = {
                "size": size,
                "average_execution_time": results[size],
            }

        json_filename = "averaged_execution_times.json"
        self.save_to_json(averaged_results, json_filename)
        print(f"Averaged results saved to {json_filename}")
    
