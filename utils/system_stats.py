import psutil
import time
import csv

# Open a new file for writing
with open('system_stats.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)', 'Network Usage (bytes)'])

    i = 1
    while i < 6:
        # Get current CPU usage
        cpu_usage = psutil.cpu_percent()

        # Get current memory usage
        memory_usage = psutil.virtual_memory().percent

        # Get current disk usage
        disk_usage = psutil.disk_usage('/').percent

        # Get current network usage
        network_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

        # Write the stats to the file
        writer.writerow([cpu_usage, memory_usage, disk_usage, network_usage])

        # Wait for 2 seconds before getting the next set of data
        time.sleep(2)
        i += 1
        
# The file will be automatically closed when the 'with' block is exited. No need to call file.close() here.