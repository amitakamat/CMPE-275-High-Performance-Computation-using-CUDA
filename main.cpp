/*
 * A program to accept an image folder, process all the images
 * and store output images in the output directory.
 * The program provides you the option to run it serially or parallely using CUDA.
 * You can perform 2 operations - Mirror an image and blur an image.
 * Time of execution is displayed at the end for performance comparison.
 * Author : Amita Vasudev Kamat
 * CMPE 275 - Project 2 - Spring 2018
 */

#include <iostream>
#include <cuda_runtime.h>
#include "load_save.h"
#include "blur_ops.h"
#include "mirror_ops.h"
#include <time.h>
#include <dirent.h> 
#include <stdio.h>
#include <rpc.h>
#include <objbase.h>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <windows.h>
#include <direct.h>  
#include <stdlib.h>  

using namespace std;

size_t numRows, numCols;

/*
 * This method reads and loads an image in GPU for parallel processing and returns the image array for serial processing
*/
uchar4* loadImage(string filename, bool isParallel)
{ 
	uchar4 *host_image, *input_image;
	
	loadImageRGBA(filename, &host_image, &numRows, &numCols);

	if (isParallel) {
		// Allocate memory to the GPU to store the image in device for parallel processing
		cudaMalloc((void **)&input_image, numRows * numCols * sizeof(uchar4));
		cudaMemcpy(input_image, host_image, numRows * numCols *  sizeof(uchar4), cudaMemcpyHostToDevice);

		// Free space
		free(host_image);
		return input_image;
	}
	else {
		return host_image;
	}
}

/*
 * Reads contents of directory and place the names in vector for input and output images.
 * Creates output directory if not present
*/
int GetDirFileNames(const std::string dir, std::vector<std::string>& files, std::vector<std::string>& ofiles, char oDirectory[]) {
	DIR* dp;
	dirent *dirp;
	if (!(dp = opendir(dir.c_str()))) {
		std::cout << "Could not open directory \"" << dir << "\"\n";
		exit(4);
	}
	int result = _mkdir(oDirectory);
	while ((dirp = readdir(dp))) {
		if (strlen(dirp->d_name) > 4) {
			string newdir(oDirectory);
			files.push_back(std::string(dir + "\\" + dirp->d_name));
			ofiles.push_back(std::string(newdir + "\\" + dirp->d_name));
		}
	}

	return 1;
}

/*
 * Main method. Starting point of execution of the program.
 */
int main(int argc, char **argv) {

	std::vector<std::string> imageName;
	std::vector<std::string> oimageName;
	char oDirectory[100] = "";
	printf("***** Program for HPC using CUDA *****\n\n");
	printf("Please select an option for processing the image \n1. Serial\n2. Parallel using CUDA\n");
	int op = 0;
	scanf("%d", &op);
	printf("*** Please select a filter for the images ***\n1. Mirror\n2. Blur\n");
	int filter = 0;
	scanf("%d", &filter);
	printf("Please enter the path of the image directory :\n");
	char dirPath[100] = "";
	scanf("%s", dirPath);
	printf("Please enter the path of the output directory :\n");
	scanf("%s", oDirectory);
	//printf("%s\n", dirPath);
	//printf("%s\n", oDirectory);
	clock_t begin = clock();
	double total = 0.0;
	int count = 1;

	uchar4 *h_out = NULL;
	int amount = 21;
	int inp_amt = 0;
	if (filter == 2) {
		printf("Please enter amount for blur (enter 0 for default value) :\n");
		scanf("%d", &inp_amt);
	}
	if (inp_amt > 0) {
		if (inp_amt % 2 == 0)
			amount = inp_amt + 1;
		else
			amount = inp_amt;
	}

	string output_file;
	if (GetDirFileNames(dirPath, imageName, oimageName, oDirectory)) {
		if (filter == 1) {
			// Selected operation - MIRROR
			begin = clock();
			if (op == 1) {
				// Selected option - SERIAL
				uchar4 *d_in;
				std::vector<std::string>::iterator inp, outp;
				for (inp = imageName.begin(), outp = oimageName.begin(); inp < imageName.end() && outp < oimageName.end(); inp++, outp++) {
					printf("Processing Image : %d\n", count);
					d_in = loadImage(*inp, false);
					output_file = *outp;
					begin = clock();
					h_out = mirror_ops(d_in, numRows, numCols, true, false);
					total += (double)(clock() - begin);
					if (h_out != NULL)
						saveImageRGBA(h_out, output_file, numRows, numCols);
					h_out = NULL;
					printf("Processing Completed!\n");
					count++;
				}
				cudaFree(d_in);
			}

			else {
				// Selected option - Parallel
				uchar4 *d_in;
				std::vector<std::string>::iterator inp, outp;
				for (inp = imageName.begin(), outp = oimageName.begin(); inp < imageName.end() && outp < oimageName.end(); inp++, outp++) {
					printf("Processing Image : %d\n", count);
					d_in = loadImage(*inp, true);
					output_file = *outp;
					begin = clock();
					h_out = mirror_ops(d_in, numRows, numCols, true, true);
					total += (double)(clock() - begin);
					if (h_out != NULL)
						saveImageRGBA(h_out, output_file, numRows, numCols);
					h_out = NULL;
					printf("Processing Completed!\n");
					count++;
				}
				cudaFree(d_in);

			}
		}
		else {
			// Selected operation - BLUR
			if (op == 1) {
				// Selected option - Serial
				uchar4 *d_in;
				std::vector<std::string>::iterator inp, outp;
				for (inp = imageName.begin(), outp = oimageName.begin(); inp < imageName.end() && outp < oimageName.end(); inp++, outp++) {
					printf("Processing Image : %d\n", count);
					d_in = loadImage(*inp, false);
					output_file = *outp;
					begin = clock();
					h_out = blur_ops(d_in, numRows, numCols, amount, false);
					total += (double)(clock() - begin);
					if (h_out != NULL)
						saveImageRGBA(h_out, output_file, numRows, numCols);
					h_out = NULL;
					printf("Processing Completed!\n");
					count++;
				}
				cudaFree(d_in);
			}
			else {
				// Selected option - Parallel
				uchar4 *d_in;
				std::vector<std::string>::iterator inp, outp;
				for (inp = imageName.begin(), outp = oimageName.begin(); inp < imageName.end() && outp < oimageName.end(); inp++, outp++) {
					printf("Processing Image : %d\n", count);
					d_in = loadImage(*inp, true);
					output_file = *outp;
					begin = clock();
					h_out = blur_ops(d_in, numRows, numCols, amount, true);
					total += (double)(clock() - begin);
					if (h_out != NULL)
						saveImageRGBA(h_out, output_file, numRows, numCols);
					h_out = NULL;
					printf("Processing Completed!\n");
					count++;
				}
				cudaFree(d_in);
			}
		}

	}
	total = total / CLOCKS_PER_SEC;
	printf("Processing of all images Completed!\n");
	printf("Processing Time : %f sec!\n", total);
	*dirPath = NULL;
	printf("Press any button to exit");
	int exit = 0;
	scanf("%d", &op);
}