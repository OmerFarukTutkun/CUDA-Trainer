#include "activations.h"

__global__ void reluKernel(float *unactivated, float *activated, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		activated[index] = MAX(0.0f, unactivated[index]);
}
__global__ void sigmoidKernel(float *unactivated, float *activated, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		activated[index] = 1.0f / (1.0f + expf(-unactivated[index]));
}
__global__ void clippedReluKernel(float *unactivated, float *activated, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		activated[index] = MAX(0.0f, MIN(ClippedReluMax, unactivated[index]));
}

__global__ void backpropReluKernel(float *unactivated, float *activated, float *gradients, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		if (unactivated[index] < 0.0f)
			gradients[index] = 0.0f;
	}
}
__global__ void backpropSigmoidKernel(float *unactivated, float *activated, float *gradients, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		gradients[index] *= activated[index] * (1 - activated[index]);
}
__global__ void backpropClippedReluKernel(float *unactivated, float *activated, float *gradients, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		if (unactivated[index] < 0 || unactivated[index] > ClippedReluMax)
			gradients[index] = 0.0f;
	}
}