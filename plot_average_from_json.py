import json
import matplotlib.pyplot as plt
import os

def plot_average_from_json(json_filename1, json_filename2):
    if not os.path.isfile(json_filename1):
        print(f"File '{json_filename1}' does not exist.")
        return

    if not os.path.isfile(json_filename2):
        print(f"File '{json_filename2}' does not exist.")
        return

    with open(json_filename1, 'r') as json_file1:
        data1 = json.load(json_file1)

    with open(json_filename2, 'r') as json_file2:
        data2 = json.load(json_file2)

    sizes = []
    average1 = []
    average2 = []

    # Retrieve data from JSON file 1
    for size, result in data1.items():
        sizes.append(int(result['size']))
        average1.append(float(result['average_execution_time']))

    # Retrieve data from JSON file 2
    for size, result in data2.items():
        average2.append(float(result['average_execution_time']))

    # Calculate the average of the two data sets
    average = [(a + b) / 2 for a, b in zip(average1, average2)]

    # Plotting the line chart
    plt.plot(sizes, average)
    plt.xlabel('Size')
    plt.ylabel('Average Execution Time')
    plt.title('Averaged Execution Times - Line Plot')

    # Save the chart as an image file
    line_chart_name = 'line_chart_AVG_CPU_GPU_matrix.png'
    plt.savefig(line_chart_name)

    # Save the averaged data as a JSON file
    data_filename = 'averaged_data_CPU_GPU_matrix.json'
    averaged_data = {
        'sizes': sizes,
        'average_execution_times': average
    }
    with open(data_filename, 'w') as json_output:
        json.dump(averaged_data, json_output)

    plt.show()
