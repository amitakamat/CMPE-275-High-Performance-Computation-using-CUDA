# CMPE 275 High Performance Computation In Image Processing using CUDA
  
## Deep dive into High Performance Computation using CUDA in Image Processing algorithms.  
  
### Image Processing Algorithms:  
1. Blur  
2. Mirror  
  
### Steps followed:  
1. Setup environment for CUDA development.  
2. Implement Mirror and blur image processing algorithms in serial and in parallel using CUDA kernel code.  
3. Calculate time taken for processing images.  
4. Test the performance of parallel code over serial code for different parameters - Blur amount, different input size.  
  
### Tools and Technologies:  
C++, Visual Studio 2017, NSight, NVIDIA GeForce GTX 1060 3GB, CUDA development tools
  
### System Configuration for testing  
Graphics Card:  NVIDIA GeForce GTX 1060 3GB  
GPU Architecture: Pascal  
CUDA Cores: 1152  
Compute Capability Version: 6.1  
Maximum number of threads per block: 1024  
OS: Windows 10  
Processor: Intel Core i7-7700k@4.20GHz  
RAM: 16GB  
Cores: 8  
  
### Performance comparison for Mirror Images  
Time is in seconds  

| Directory Size | Total Files | Serial Time | Parallel Time | % gain|  
|--------------- |:----------: | :---------: | :-----------: | :----: |  
| 4KB            | 1           | 0.001       | 0.001         | 0      |
| 10MB           | 13          | 0.079       | 0.039         | 50.63  |
| 70MB           | 15          | 0.383       | 0.146         | 61.879 |
| 267MB          | 21          | 1.70        | 0.482         | 71.64  |
| 1GB            | 80          | 7.836       | 2.001         | 74.46  |
  
   
### Performance comparison for Blur Images with constant blur amount 11  
Time is in seconds   

| Directory Size | Total Files | Serial Time | Parallel Time | % gain|  
|--------------- |:----------: | :---------: | :-----------: | :----: |  
| 4KB            | 1           | 0.403       | 0.046         | 88.58  |
| 10MB           | 13          | 45.72       | 2.551         | 94.42  |
| 70MB           | 15          | 318.602     | 15.90         | 95     |
| 267MB          | 21          | 1637.9      | 80            | 95.11  |
| 1GB            | 80          | 7163.368    | 343.914       | 95.19  |  
  
  
### Performance comparison for Blur Images with constant input size 10MB  
Time is in seconds  

| Blur Amount | Total Files | Serial Time | Parallel Time | % gain|  
|------------ |:----------: | :---------: | :-----------: | :----: |  
| 11          | 13          | 45.72       | 2.551         | 94.42  |
| 31          | 13          | 357.952     | 17.549        | 95.09  |
| 40          | 15          | 625.305     | 30.531        | 95.117 |
  
  
### Input Image:  
![Original Input Image](/images/original.jpg "Original Image")
  
### Mirror Image: 
![Mirror Image](/images/mirror.png "Mirror Image")

### Blur Image with Blur amount 11: 
![Blur Image with Blur amount 11](/images/blur11.png "Blur Image")

### Blur Image with Blur amount 31: 
![Blur Image with Blur amount 31](/images/blur31.png "Blur Image")

### Blur Image with Blur amount 40: 
![Blur Image with Blur amount 40](/images/blur%2040.png "Blur Image")

### Program output for mirror using serial code: 
![mirror serial](/images/Mirror-serial-output.png "Serial Mirror")

### Program output for blur using parallel code: 
![blur parallel](/images/Blur-parallel-output.png "Blur Parallel")
