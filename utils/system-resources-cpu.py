import psutil
import csv

with open('system-resources-cpu.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['system-resources (%)'])

    psutil.cpu_percent(interval=1)
    per_cpu = psutil.cpu_percent(percpu=True)
    # For individual core usage with blocking, psutil.cpu_percent(interval=1, percpu=True)
    for idx, usage in enumerate(per_cpu):
        writer.writerow([f"CORE_{idx+1}: {usage}%"])