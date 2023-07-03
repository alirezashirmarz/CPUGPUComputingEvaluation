# ****************************************** written by Alireza ***************************************
import cv2
import time

# Load an image for processing
image_path = "path_to_your_image.jpg"
image = cv2.imread(image_path)

# CPU Image Processing
def cpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using CPU
    # Example: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time

# GPU Image Processing
def gpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using GPU
    # Example: Convert the image to grayscale
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)
    gpu_gray_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
    gray_image = gpu_gray_image.download()

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time

# Run CPU Image Processing
cpu_result, cpu_execution_time = cpu_image_processing(image)

# Run GPU Image Processing
gpu_result, gpu_execution_time = gpu_image_processing(image)

# Display the results and execution times
cv2.imshow("CPU Result", cpu_result)
cv2.imshow("GPU Result", gpu_result)
cv2.waitKey(0)

print("CPU Execution Time:", cpu_execution_time, "seconds")
print("GPU Execution Time:", gpu_execution_time, "seconds")

""" Code Definition to use: 
In this code, we start by loading an image for processing using the cv2.imread function. 
Then, we define two functions: cpu_image_processing and gpu_image_processing, 
which perform image processing operations using CPU and GPU, respectively. 

In the example, we convert the image to grayscale using the cv2.cvtColor function 
for both CPU and GPU processing.

The cpu_image_processing function measures the execution time using the time module. 
Similarly, the gpu_image_processing function measures the execution time 
but with GPU-accelerated operations using the OpenCV CUDA module.

After running the CPU and GPU image processing functions, 
the results are displayed using cv2.imshow, and the execution times are printed.

Please note that for this code to work, you need to have OpenCV installed with CUDA support. 
Additionally, you may need to modify the image processing operations based on 
your specific requirements.

Remember to replace "path_to_your_image.jpg" with the actual path to your image file 
"""
