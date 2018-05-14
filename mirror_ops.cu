/*
* This file contains the following:
* Mirroring functionality serially
* Mirroring functionality parallely using CUDA
* Author : Amita Vasudev Kamat
* CMPE 275 - Project 2 - Spring 2018
*/

#include <cuda_runtime.h>
#include <stdio.h>    
#include "device_launch_parameters.h"
#include <Windows.h>
#include <stdlib.h>

/* Mirror operations */
uchar4* d_outRGBA;

/* 
* Kernel for mirroring the image parallely usng CUDA
*/
__global__ 
void mirror(const uchar4* const inputChannel, uchar4* outputChannel, int numRows, int numCols, bool vertical)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ( col >= numCols || row >= numRows )
  {
   return;
  }

  if(!vertical)
  { 
  
    int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int thread_x_new = thread_x;
    int thread_y_new = numRows-thread_y;

    long myId = thread_y * numCols + thread_x;
    long myId_new = thread_y_new * numCols + thread_x_new;
    outputChannel[myId_new] = inputChannel[myId];
   	
  }

  else
  {
	  unsigned int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
	  unsigned int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
    
	  unsigned int thread_x_new = numCols-thread_x;
	  unsigned int thread_y_new = thread_y;

	  unsigned long int myId = thread_y * numCols + thread_x;
	  unsigned long int myId_new = thread_y_new * numCols + thread_x_new;
	//printf("Id : %lu\t NewId : %lu\n", myId, myId_new);
  
  	outputChannel[myId_new] = inputChannel[myId];  // linear data store in global memory	
  }
}     

/*
* Method for mirroring the image serially
*/
void mirror_serial(const uchar4* const inputChannel, uchar4* outputChannel, int numRows, int numCols, bool vertical)
{
	

	if (!vertical)
	{

	}

	else
	{
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {

				int x_new = numCols - j;
				int y_new = i;

				int myId = i * numCols + j;
				int myId_new = y_new * numCols + x_new;

				outputChannel[myId_new] = inputChannel[myId];  // linear data store in global memory	
			}
		}
	}
}

/*
* Method to calculate grid size for the image and allocate required memory for parallel processing.
*/
uchar4* mirror_ops(uchar4 *d_inputImageRGBA, size_t numRows, size_t numCols, bool vertical, bool isParallel)
{
	cudaError_t cudaStatus;
	//Set number of threads per block)
	const dim3 blockSize(16, 16);
	//const int blockSize = 250;
	  //Calculate Grid SIze
	  int a=numCols/ blockSize.x, b=numRows/ blockSize.y;
	  const dim3 gridSize(a+1,b+1,1);

  const size_t numPixels = numRows * numCols;
  //Initialize memory on host for output uchar4*
  uchar4* h_out;
  h_out = (uchar4*)malloc(sizeof(uchar4) * numPixels);


  if (isParallel) {
	  uchar4 *d_outputImageRGBA;
	  cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels);

	  //Call mirror kernel.
	  mirror << <gridSize, blockSize >> > (d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, vertical);

	  cudaStatus = cudaDeviceSynchronize();
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching addKernel!\n", cudaGetErrorString(cudaStatus));
	  }

	  //copy output from device to host
	  cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

	Error:
	  //cleanup memory on device
	  cudaFree(d_inputImageRGBA);
	  cudaFree(d_outputImageRGBA);
  }
  else {
	  d_outRGBA = (uchar4*)malloc(sizeof(uchar4) * numPixels);
	  mirror_serial(d_inputImageRGBA, d_outRGBA, numRows, numCols, vertical);
	  h_out = d_outRGBA;
  }

	return h_out;
}