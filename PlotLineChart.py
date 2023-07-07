import json
import matplotlib.pyplot as plt
import os

def plot_line_chart_from_json(json_filename):
    if not os.path.isfile(json_filename):
        print(f"File '{json_filename}' does not exist.")
        return

    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)

    sizes = []
    execution_times = []

    if json_filename=='averaged_execution_cpu_times.json':
        lineChart_name = "lineChart_matrix_cpu.png"   
        lineColor = 'red'   
    else:
       lineChart_name = "lineChart_matrix_gpu.png"    
       lineColor = 'blue'

    for size, result in data.items():
        sizes.append(int(result['size']))
        execution_times.append(float(result['average_execution_time']))

    plt.plot(sizes, execution_times,color=lineColor)
    plt.xlabel('Size')
    plt.ylabel('Average Execution Time')
    plt.title('Averaged Execution Times - Line Plot')
    plt.savefig(lineChart_name)
    plt.show()

