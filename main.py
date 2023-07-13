import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import cupy as cp
import psutil
import GPUtil as GPU

print('The CPU usage is: ', psutil.cpu_percent(1))
print('RAM memory % used:', psutil.virtual_memory()[2])
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

# Scenario 1 

average_cpu_matrix = 0
average_cpu_matrix_list = []

average_gpu_matrix = 0
average_gpu_matrix_list = []

matrix_sizes = [100, 200, 400, 700, 10000]

def cpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    cpu_result = np.dot(matrix_a, matrix_b)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

def gpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    gpu_matrix_a = cp.asarray(matrix_a)
    gpu_matrix_b = cp.asarray(matrix_b)
    gpu_result = cp.dot(gpu_matrix_a, gpu_matrix_b)
    cpu_result = cp.asnumpy(gpu_result)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

for matrix_size in matrix_sizes:

    for i in range(5):
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        cpu_result, cpu_execution_time = cpu_matrix_multiplication(matrix_a, matrix_b)
        print(i , " Run(s) " , "CPU Execution Time:", cpu_execution_time, "seconds")
        #print(i , " Run(s) " , "CPU Result:")
        #print(cpu_result)
        average_cpu_matrix = average_cpu_matrix + cpu_execution_time

    average_cpu_matrix_list.append(average_cpu_matrix)
    print("Matrix Size: " , matrix_size , " || Average CPU Execution Time:" , average_cpu_matrix / 5, "seconds")

print('The CPU usage is: ', psutil.cpu_percent(1))
print('RAM memory % used:', psutil.virtual_memory()[2])
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

plt.plot(matrix_sizes, average_cpu_matrix_list)
plt.xlabel('Matrix Size')
plt.ylabel('Average CPU Execution Time')
plt.title('CPU Matrix')
plt.savefig('cpu_matrix_multiplication.png')
plt.show()

GPUs = GPU.getGPUs()
for gpu in GPUs:
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))


for matrix_size in matrix_sizes:

    for i in range(5):
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        gpu_result, gpu_execution_time = gpu_matrix_multiplication(matrix_a, matrix_b)
        print(i , " Run(s) " , "GPU Execution Time:" , gpu_execution_time , "seconds")
        #print(i , " Run(s) " , "GPU Result:")
        #print(gpu_result)
        average_gpu_matrix = average_gpu_matrix + gpu_execution_time

    average_gpu_matrix_list.append(average_gpu_matrix)
    print("Matrix Size: " , matrix_size , " || Average GPU Execution Time:", average_gpu_matrix / 5, "seconds")

GPUs = GPU.getGPUs()
for gpu in GPUs:
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

plt.plot(matrix_sizes, average_gpu_matrix_list)
plt.xlabel('Matrix Size')
plt.ylabel('Average GPU Execution Time')
plt.title('GPU Matrix')
plt.savefig('gpu_matrix_multiplication.png')
plt.show()

# Scenario 2 

""" average_cpu_image = 0
average_cpu_image_list = []

average_gpu_image = 0
average_gpu_image_list = []

image_path = "/home/massoud/aos/pics/heavyrain.jpg"
image = cv2.imread(image_path)

def cpu_image_processing(image):
    start_time = time.time()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time

def gpu_image_processing(image):
    start_time = time.time()

    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)
    gpu_gray_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
    gray_image = gpu_gray_image.download()

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time

cpu_result, cpu_execution_time = cpu_image_processing(image)

gpu_result, gpu_execution_time = gpu_image_processing(image)

cv2.imshow("CPU Result", cpu_result)
cv2.imshow("GPU Result", gpu_result)
cv2.waitKey(0)

print("CPU Execution Time:", cpu_execution_time, "seconds")
print("GPU Execution Time:", gpu_execution_time, "seconds") """


#window = tk.Tk()

#greeting = tk.Label(text="Calculation Metrics")

#greeting.pack()

#window.mainloop()

