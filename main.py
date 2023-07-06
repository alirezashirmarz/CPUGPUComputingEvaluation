import json
from utils.cpu_operations import cpu_operations
from utils.gpu_operations import gpu_operations
import BarChartPlotter
import PlotLineChart
import matplotlib.pyplot as plt

def main():

    # Create an instance of the cpu_operations class
    cpu_operat = cpu_operations()
    gpu_ooerat = gpu_operations()
   
    # Perform matrix multiplication on CPU
    
    cpu_operat.calculate_and_save_averaged_results()
    gpu_ooerat.calculate_and_save_averaged_results()
    
    
    # show BarChartPlotter
    json_filename_cpu = "averaged_execution_cpu_times.json"
    json_filename_gpu = "averaged_execution_gpu_times.json"
    #BarChartPlotter.plot_bar_chart_from_json(json_filename)
    PlotLineChart.plot_line_chart_from_json(json_filename_cpu)
    PlotLineChart.plot_line_chart_from_json(json_filename_gpu)

if __name__ == '__main__':
    main()