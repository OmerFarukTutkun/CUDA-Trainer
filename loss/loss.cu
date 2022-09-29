#include "loss.h"

Loss MSE = {mse , backpropMse};
Loss MAE = {mae , backpropMae};

void mse(Matrix *prediction, Matrix *target, Matrix *loss)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(loss));
	assert(checkDimension(prediction, target) && checkDimension(loss, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	mseKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, loss->data, size);

	cudaDeviceSynchronize();
}
void mae(Matrix *prediction, Matrix *target, Matrix *loss)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(loss));
	assert(checkDimension(prediction, target) && checkDimension(loss, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	maeKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, loss->data, size);

	cudaDeviceSynchronize();
}

void backpropMse(Matrix *prediction, Matrix *target, Matrix *lossGradient)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(lossGradient));
	assert(checkDimension(prediction, target) && checkDimension(lossGradient, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	backpropMseKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, lossGradient->data, size);

	cudaDeviceSynchronize();
}
void backpropMae(Matrix *prediction, Matrix *target, Matrix *lossGradient)
{
	assert(checkMemory(prediction) && checkMemory(target) && checkMemory(lossGradient));
	assert(checkDimension(prediction, target) && checkDimension(lossGradient, target));

	int size = prediction->rows * prediction->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	backpropMaeKernel<<<nBlocks, BlockSize>>>(prediction->data, target->data, lossGradient->data, size);

	cudaDeviceSynchronize();
}