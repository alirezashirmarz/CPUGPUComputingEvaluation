import json
import matplotlib.pyplot as plt
import os

def plot_bar_chart_from_json(json_filename):
    if not os.path.isfile(json_filename):
        print(f"File '{json_filename}' does not exist.")
        return

    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)

    sizes = []
    execution_times = []

    for size, result in data.items():
        sizes.append(int(result['size']))
        execution_times.append(float(result['average_execution_time']))

    plt.bar(sizes, execution_times)
    plt.xlabel('Size')
    plt.ylabel('Average Execution Time')
    plt.title('Averaged Execution Times')
    plt.savefig('bar_chart.png')
    plt.show()


