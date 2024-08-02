#include "loss.h"
constexpr float K = 2.65;
__global__ void mseKernel(float *prediction, float *target, float *loss, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
            loss[index] = loss[index] + powf( fabs(target[index] - prediction[index]) , K);
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
		lossGradient[index] = -K * (target[index] - prediction[index]) * powf( fabs(target[index] - prediction[index]) , K - 2.0);;
}
__global__ void backpropMaeKernel(float *prediction, float *target, float *lossGradient, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
		lossGradient[index] = target[index] > prediction[index] ? -1.0 : 1.0;
}
