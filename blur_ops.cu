/*
* This file contains the following:
* Method to blur image parallely using CUDA(RGB in parallel)
* Method to blur image parallely using CUDA
* Method to blur image serially
* Author : Amita Vasudev Kamat
* CMPE 275 - Project 2 - Spring 2018
*/

#include <cuda_runtime.h>
#include <stdio.h>    
#include <Windows.h>
#include <stdlib.h>
#include <math.h>
#include "device_launch_parameters.h"

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;
uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;
float *h_filter__;
uchar4* dblur_outRGBA;

/*
* Kernel for blurring the image parallely usng CUDA- general
*/
__global__
void gaussian_blur_1(const uchar4* const inputChannel, uchar4* outputChannel,
	int numRows, int numCols, const float* const filter, const int filterWidth)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= numCols || row >= numRows)
	{
		return;
	}

	long myId = row * numCols + col;
	float result_x = 0.f;
	float result_y = 0.f;
	float result_z = 0.f;
	
	for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; filter_r++)
	{

		for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; filter_c++)
		{

			float image_value_x = static_cast<float>(inputChannel[myId].x);
			float filter_value = filter[(filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2];
			result_x += image_value_x * filter_value;

			float image_value_y = static_cast<float>(inputChannel[myId].y);
			result_y += image_value_y * filter_value;

			float image_value_z = static_cast<float>(inputChannel[myId].z);
			result_z += image_value_z * filter_value;

		}
	}
	uchar4 pix = make_uchar4(result_x, result_y, inputChannel[myId].z, 255);
	outputChannel[row * numCols + col] = pix;
}

/*
* Kernel for blurring the image parallely usng CUDA- RGB in parallel
*/
__global__
void gaussian_blur(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= numCols || row >= numRows) {
		return;
	}

	float c = 0.0f;

	for (int fx = 0; fx < filterWidth; fx++) {
		for (int fy = 0; fy < filterWidth; fy++) {
			int imagex = col + fx - filterWidth / 2;
			int imagey = row + fy - filterWidth / 2;
			imagex = min(max(imagex, 0), numCols - 1);
			imagey = min(max(imagey, 0), numRows - 1);
			c += (filter[fy*filterWidth + fx] * inputChannel[imagey*numCols + imagex]);
		}
	}

	outputChannel[row*numCols + col] = c;
}

/*
* Method for blurring the image serially
*/
void gaussian_blur_serial(const uchar4* const inputChannel, uchar4* outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth) {
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			float x = 0.0f;
			float y = 0.0f;
			float z = 0.0f;

			for (int fx = 0; fx < filterWidth; fx++) {
				for (int fy = 0; fy < filterWidth; fy++) {
					int imagex = j + fx - filterWidth / 2;
					int imagey = i + fy - filterWidth / 2;
					imagex = min(max(imagex, 0), numCols - 1);
					imagey = min(max(imagey, 0), numRows - 1);
					x += (filter[fy*filterWidth + fx] * static_cast<float>(inputChannel[imagey*numCols + imagex].x));
					y += (filter[fy*filterWidth + fx] * static_cast<float>(inputChannel[imagey*numCols + imagex].y));
					z += (filter[fy*filterWidth + fx] * static_cast<float>(inputChannel[imagey*numCols + imagex].z));
				}
			}
			uchar4 pix = make_uchar4(x, y, z, 255);
			outputChannel[i*numCols + j] = pix;
		}
	}

}

/*
* Kernel to separate RGB channels
*/
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{  
  int absolute_image_position_x = blockDim.x * blockIdx.x + threadIdx.x;
  int absolute_image_position_y = blockDim.y * blockIdx.y + threadIdx.y;

  if ( absolute_image_position_x >= numCols ||
      absolute_image_position_y >= numRows )
  {
       return;
  }
  
  int thread_1D_pos = absolute_image_position_y * numCols + absolute_image_position_x;

  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

/*
* Kernel for combining RGB channels
*/
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

/*
* Method to free the space
*/
void cleanup() {
	cudaFree(d_red);
	cudaFree(d_green);
	cudaFree(d_blue);
	cudaFree(d_filter);
}


/*
* Method to allocate memory for RGB pixels in GPU
*/
void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage, const float* const h_filter, const size_t filterWidth)
{ 

	cudaError_t cudaStatus;
	cudaStatus =  cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cleanup();
	}
	cudaStatus = cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cleanup();
	}
	cudaStatus = cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cleanup();
	}

  //Allocate memory for the filter on the GPU
	cudaStatus =  cudaMalloc(&d_filter, sizeof(float)*filterWidth*filterWidth);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cleanup();
	}
	cudaStatus = cudaMemcpy(d_filter,h_filter,sizeof(float)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cleanup();
	}
}

/*
* Method to aset the filter for blur
*/
void setFilter(float **h_filter, int *filterWidth, int blurKernelWidth, float blurKernelSigma)
{  
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  *filterWidth = blurKernelWidth;
  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r)
   {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) 
    {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;
  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) 
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) 
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
}

/* 
 * Method to calculate filter, grid size and process the image
*/
uchar4* blur_ops(uchar4* d_inputImageRGBA, size_t numRows, size_t numCols, int blurKernelWidth, bool isParallel)
{ 
	cudaError_t cudaStatus;
	float blurKernelSigma = blurKernelWidth/4.0f;
	  //Set filter array
	  float* h_filter;
	  const size_t numPixels = numRows * numCols;
	  int filterWidth;
	  setFilter(&h_filter, &filterWidth, blurKernelWidth, blurKernelSigma);
	  uchar4* h_out;
	  h_out = (uchar4*)malloc(sizeof(uchar4) * numPixels);

  if (isParallel) {

	  const dim3 blockSize(16,16,1);
	  //Calculate Grid Size
	  int a=numCols/blockSize.x, b=numRows/blockSize.y;	
	  const dim3 gridSize(a+1,b+1,1);


	  // Choose which GPU to run on, change this on a multi-GPU system.
	  cudaStatus = cudaSetDevice(0);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		  goto Error;
	  }

	  uchar4 *d_outputImageRGBA;
	  cudaStatus = cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaMalloc failed!");
		  goto Error;
	  }
	  cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4)); //make sure no memory is left laying around

	  d_inputImageRGBA__  = d_inputImageRGBA;
	  d_outputImageRGBA__ = d_outputImageRGBA;


	  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
	  cudaStatus = cudaMalloc(&d_redBlurred, sizeof(unsigned char) * numPixels);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaMalloc failed!");
		  goto Error;
	  }
	  cudaStatus = cudaMalloc(&d_greenBlurred, sizeof(unsigned char) * numPixels);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaMalloc failed!");
		  goto Error;
	  }
	  cudaStatus = cudaMalloc(&d_blueBlurred, sizeof(unsigned char) * numPixels);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaMalloc failed!");
		  goto Error;
	  }
	  cudaStatus = cudaMemset(d_redBlurred, 0, sizeof(unsigned char) * numPixels);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaMemset failed!");
		  goto Error;
	  }
	  cudaStatus = cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaMemset failed!");
		  goto Error;
	  }
	  cudaStatus = cudaMemset(d_blueBlurred, 0, sizeof(unsigned char) * numPixels);
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaMemset failed!");
		  goto Error;
	  }

	  allocateMemoryAndCopyToGPU(numRows, numCols, h_filter, filterWidth);

	  //size_t shm_size = 16 * 16 * sizeof(unsigned long long);
	  separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

	  cudaStatus = cudaDeviceSynchronize();
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaDeviceSynchronize1 returned error code %d after launching addKernel!\n", cudaStatus);
		  goto Error;
	  }

	  gaussian_blur << <gridSize, blockSize >> > (d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
	  gaussian_blur << <gridSize, blockSize >> > (d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
	  gaussian_blur << <gridSize, blockSize >> > (d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

	  cudaStatus = cudaDeviceSynchronize();
	  if (cudaStatus != cudaSuccess) {
		  //fprintf(stderr, "cudaDeviceSynchronize2 returned error code %d after launching addKernel!\n", cudaStatus);
		  fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(cudaStatus));
		  goto Error;
	  }

	  //Combine the results.
	  recombineChannels << <gridSize, blockSize >> > (d_redBlurred,
		  d_greenBlurred,
		  d_blueBlurred,
		  d_outputImageRGBA,
		  numRows,
		  numCols);
	  cudaStatus = cudaDeviceSynchronize();
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaDeviceSynchronize3 returned error code %d after launching addKernel!\n", cudaStatus);
		  goto Error;
	  }
	  //cleanup memory

	  cudaDeviceSynchronize();

	  //copy output from device to host
	  cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

	  cudaDeviceSynchronize();

  Error:
	  cleanup();
	  cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
	  cudaDeviceSynchronize();
	  cudaFree(d_inputImageRGBA__);
	  cudaFree(d_outputImageRGBA__);
	  cudaFree(d_outputImageRGBA);
	  cudaFree(d_redBlurred);
	  cudaFree(d_greenBlurred);
	  cudaFree(d_blueBlurred);
	  cudaFree(d_filter);
	  delete[] h_filter__;

	  cudaStatus = cudaDeviceReset();
	  if (cudaStatus != cudaSuccess) {
		  fprintf(stderr, "cudaDeviceReset failed!");
	  }
  }
  else {
	  dblur_outRGBA = (uchar4*)malloc(sizeof(uchar4) * numPixels);
	  gaussian_blur_serial(d_inputImageRGBA, dblur_outRGBA, numRows, numCols, h_filter, filterWidth);
	  h_out = dblur_outRGBA;
  }

	return h_out;
}