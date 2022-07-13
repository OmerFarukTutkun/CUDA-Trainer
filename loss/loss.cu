#include "loss.h"
void MSE(Matrix *prediction, Matrix *target, Matrix *loss)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(loss));
	assert(checkDimension(prediction, target) && checkDimension(loss, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	MSEKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, loss->data, size);

	cudaDeviceSynchronize();
}
void MAE(Matrix *prediction, Matrix *target, Matrix *loss)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(loss));
	assert(checkDimension(prediction, target) && checkDimension(loss, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	MAEKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, loss->data, size);

	cudaDeviceSynchronize();
}

void backpropMSE(Matrix *prediction, Matrix *target, Matrix *lossGradient)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(lossGradient));
	assert(checkDimension(prediction, target) && checkDimension(lossGradient, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	backpropMSEKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, lossGradient->data, size);

	cudaDeviceSynchronize();
}
void backpropMAE(Matrix *prediction, Matrix *target, Matrix *lossGradient)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(lossGradient));
	assert(checkDimension(prediction, target) && checkDimension(lossGradient, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	backpropMAEKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, lossGradient->data, size);

	cudaDeviceSynchronize();
}