/*
* Header for mirror operation method
* Author : Amita Vasudev Kamat
* CMPE 275 - Project 2 - Spring 2018
*/

#include <cuda_runtime.h>
#ifndef MIRROR_OPS_H__
#define MIRROR_OPS_H__

uchar4* mirror_ops(uchar4 *d_inputImageRGBA, size_t numRows, size_t numCols, bool vertical, bool isParallel);  

#endif