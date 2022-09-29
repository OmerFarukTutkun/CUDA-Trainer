#include "loss.h"
__global__ void mseKernel(float *prediction, float *target, float *loss, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		loss[index] = loss[index] + (target[index] - prediction[index]) * (target[index] - prediction[index]);
}
__global__ void maeKernel(float *prediction, float *target, float *loss, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		loss[index] = loss[index] + fabsf(target[index] - prediction[index]);
}

__global__ void backpropMseKernel(float *prediction, float *target, float *lossGradient, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		lossGradient[index] = -2.0 * (target[index] - prediction[index]);
}
__global__ void backpropMaeKernel(float *prediction, float *target, float *lossGradient, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		lossGradient[index] = target[index] > prediction[index] ? -1.0 : 1.0;
}
