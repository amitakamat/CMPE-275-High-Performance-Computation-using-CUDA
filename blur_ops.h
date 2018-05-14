/*
* Header for blurring operation method
* Author : Amita Vasudev Kamat
* CMPE 275 - Project 2 - Spring 2018
*/

#include <cuda_runtime.h>
#ifndef BLUR_OPS_H__
#define BLUR_OPS_H__

void setFilter(float **h_filter, int *filterWidth, int blurKernelWidth, float blurKernelSigma);
uchar4* blur_ops(uchar4* d_in, size_t numRows, size_t numCols, int blurKernelWidth, bool isParallel);

#endif
