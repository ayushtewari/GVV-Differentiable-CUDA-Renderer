#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include "../Utils/cuda_SimpleMatrixUtil.h"

__global__ void gradientImageKernel(float3 *input, float3 *gradX, float3 *gradY, int width, int height)
{
	const unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col < 1 || row < 1 || col >= width - 2 || row >= height - 2)
	{
		gradX[row * width + col] = make_float3(0.0, 0.0, 0.0);
		gradY[row * width + col] = make_float3(0.0, 0.0, 0.0);
		return;
	}

	float3 I_00 = input[(row - 1) * width + (col - 1)];
	float3 I_01 = input[(row - 1) * width + (col + 0)];
	float3 I_02 = input[(row - 1) * width + (col + 1)];
	float3 I_10 = input[(row + 0) * width + (col - 1)];
	float3 I_12 = input[(row + 0) * width + (col + 1)];
	float3 I_20 = input[(row + 1) * width + (col - 1)];
	float3 I_21 = input[(row + 1) * width + (col + 0)];
	float3 I_22 = input[(row + 1) * width + (col + 1)];

	gradX[row * width + col] = 0.125*((I_02 - I_00) + 2 * (I_12 - I_10) + (I_22 - I_20));
	gradY[row * width + col] = 0.125*((I_20 - I_00) + 2 * (I_21 - I_01) + (I_22 - I_02));

}

__global__ void gaussianImageKernel(float3 *input, float3 *output, int width, int height)
{
	const unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col < 1 || row < 1 || col > width - 2 || row > height - 2) return;

	float3 I_00 = input[(row - 1) * width + (col - 1)];
	float3 I_01 = input[(row - 1) * width + (col + 0)];
	float3 I_02 = input[(row - 1) * width + (col + 1)];
	float3 I_10 = input[(row + 0) * width + (col - 1)];
	float3 I_11 = input[(row + 0) * width + (col + 0)];
	float3 I_12 = input[(row + 0) * width + (col + 1)];
	float3 I_20 = input[(row + 1) * width + (col - 1)];
	float3 I_21 = input[(row + 1) * width + (col + 0)];
	float3 I_22 = input[(row + 1) * width + (col + 1)];

	output[row * width + col] = 0.0625*(I_00 + I_02 + I_20 + I_22) + 0.125*(I_01 + I_10 + I_12 + I_21) + 0.25 * I_11;
}




extern "C" void gradientImage(float3 *frame, float3 *gradX, float3 *gradY, int width, int height)
{
	const int T_PER_BLOCK = 16;
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gradientImageKernel << <gridSize, blockSize >> >(frame, gradX, gradY, width, height);

	cutilSafeCall(cudaDeviceSynchronize());
	cutilSafeCall(cudaGetLastError());
}
