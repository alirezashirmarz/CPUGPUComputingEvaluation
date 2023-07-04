import json
from utils.cpu_operations import cpu_operations
import BarChartPlotter
import matplotlib.pyplot as plt

def main():

    # Create an instance of the cpu_operations class
    cpu_operat = cpu_operations()
   
    # Perform matrix multiplication on CPU
    
    cpu_operat.calculate_and_save_averaged_results()
    
    
    # show BarChartPlotter
    json_filename = "averaged_execution_times.json"
    BarChartPlotter.plot_bar_chart_from_json(json_filename)

if __name__ == '__main__':
    main()