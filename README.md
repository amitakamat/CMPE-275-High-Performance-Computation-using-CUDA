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
