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

import cv2
import numpy as np
import time
import cupy as cp

# Load an image for processing
image_paths = [
    "drive/MyDrive/3.jpg",
    "drive/MyDrive/1.jpg",
    "drive/MyDrive/bigImage.png"
]

test_count = 50

# CPU Image Processing
def cpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using CPU
    # Example: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time

def cpu_image_processing_numpy(image):
    start_time = time.time()

    second = np.array([0.299, 0.587, 0.114])
    gray_image = np.dot(image, second)


    end_time = time.time()

    # end_time = time.time()

    execution_time = end_time - start_time
    return gray_image, execution_time

# GPU Image Processing
def gpu_image_processing_cv2_cuda(image):
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


def gpu_image_processing_cupy(image):
    # start_time = time.time()

    # gray_image = np.dot(image, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    gpu_image = cp.asarray(image)
    second = cp.asarray([0.299, 0.587, 0.114])

    start_time = time.time()


    gpu_result = cp.dot(gpu_image, second)
    x = gpu_result[0][0]

    end_time = time.time()


    gray_image = cp.asnumpy(gpu_result)

    # end_time = time.time()

    execution_time = end_time - start_time
    return gray_image, execution_time


def cpu_test(image):
    test_result = []

    for i in range(test_count):
        result, execution_time = cpu_image_processing(image)
        test_result.append(execution_time)

    return test_result

def cpu_test_numpy(image):
    test_result = []

    for i in range(test_count):
        result, execution_time = cpu_image_processing_numpy(image)
        test_result.append(execution_time)

    return test_result

def gpu_test_cupy(image):
    test_result = []

    for i in range(test_count):
        result, execution_time = gpu_image_processing_cupy(image)
        test_result.append(execution_time)

    return test_result

def gpu_test_cuda(image):
    test_result = []

    for i in range(test_count):
        result, execution_time = gpu_image_processing_cv2_cuda(image)
        test_result.append(execution_time)

    return test_result

for image_path in image_paths:
  print("Testing image ", image_path)

  image = cv2.imread(image_path)

  print("Image size is ", image.shape)

  # Run CPU Image Processing
  # cpu_result, cpu_execution_time = cpu_image_processing(image)
  cpu_test_result = cpu_test(image)
  cpu_test_result_numpy = cpu_test_numpy(image)

  # Run GPU Image Processing
  # gpu_result, gpu_execution_time = gpu_image_processing_2(image)
  gpu_test_result_cupy = gpu_test_cupy(image)
  # gpu_test_result_cuda = gpu_test_cuda(image)

  # print(cpu_result)
  # print(gpu_result)

  # Display the results and execution times
  # cv2.imshow("CPU Result", cpu_result)
  # cv2.imshow("GPU Result", gpu_result)
  # cv2.waitKey(0)

  print("CPU Execution Time:              ", cpu_test_result, "seconds")
  print("CPU Execution Time (numPy):      ", cpu_test_result_numpy, "seconds")
  print("GPU Execution Time (CuPy):       ", gpu_test_result_cupy, "seconds")
  # print("GPU Execution Time (CV2_CUDA):   ", gpu_test_result_cuda, "seconds")

  print("\n")

  print("CPU Execution Time Average:              ", sum(cpu_test_result) / test_count, "seconds")
  print("CPU Execution Time Average (NumPy):      ", sum(cpu_test_result_numpy) / test_count, "seconds")
  print("GPU Execution Time Average (CuPy):       ", sum(gpu_test_result_cupy) / test_count, "seconds")
  # print("GPU Execution Time Average (CUDA):       ", sum(gpu_test_result_cuda) / test_count, "seconds")

  print("#################\n\n\n")


