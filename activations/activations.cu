#include "activations.h"

Activation Relu = {relu, backpropRelu};
Activation ClippedRelu = {clippedRelu, backpropClippedRelu};
Activation Sigmoid = {sigmoid, backpropSigmoid};
Activation SquaredClippedRelu = {squaredClippedRelu, backpropSquaredClippedRelu};

void relu(Matrix *unactivated, Matrix *activated)
{
	assert(checkMemory(unactivated) && checkMemory(activated));
	assert(checkDimension(unactivated, activated));

	int size = unactivated->rows * unactivated->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	reluKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, size);

	cudaDeviceSynchronize();
}
void sigmoid(Matrix *unactivated, Matrix *activated)
{
	assert(checkMemory(unactivated) && checkMemory(activated));
	assert(checkDimension(unactivated, activated));

	int size = unactivated->rows * unactivated->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	sigmoidKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, size);

	cudaDeviceSynchronize();
}
void clippedRelu(Matrix *unactivated, Matrix *activated)
{
	assert(checkMemory(unactivated) && checkMemory(activated));
	assert(checkDimension(unactivated, activated));

	int size = unactivated->rows * unactivated->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	clippedReluKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, size);

	cudaDeviceSynchronize();
}
void squaredClippedRelu(Matrix *unactivated, Matrix *activated)
{
    assert(checkMemory(unactivated) && checkMemory(activated));
    assert(checkDimension(unactivated, activated));

    int size = unactivated->rows * unactivated->columns;
    int nBlocks = (size - 1) / BlockSize + 1;

    squaredClippedReluKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, size);

    cudaDeviceSynchronize();
}

void backpropRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients)
{
	assert(checkMemory(unactivated) && checkMemory(activated) && checkMemory(gradients));
	assert(checkDimension(unactivated, activated) && checkDimension(gradients, activated));

	int size = unactivated->rows * unactivated->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	backpropReluKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, gradients->data, size);

	cudaDeviceSynchronize();
}
void backpropSigmoid(Matrix *unactivated, Matrix *activated, Matrix *gradients)
{
	assert(checkMemory(unactivated) && checkMemory(activated) && checkMemory(gradients));
	assert(checkDimension(unactivated, activated) && checkDimension(gradients, activated));

	int size = unactivated->rows * unactivated->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	backpropSigmoidKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, gradients->data, size);

	cudaDeviceSynchronize();
}
void backpropClippedRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients)
{
	assert(checkMemory(unactivated) && checkMemory(activated) && checkMemory(gradients));
	assert(checkDimension(unactivated, activated) && checkDimension(gradients, activated));

	int size = unactivated->rows * unactivated->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	backpropClippedReluKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, gradients->data, size);

	cudaDeviceSynchronize();
}
void backpropSquaredClippedRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients)
{
    assert(checkMemory(unactivated) && checkMemory(activated) && checkMemory(gradients));
    assert(checkDimension(unactivated, activated) && checkDimension(gradients, activated));

    int size = unactivated->rows * unactivated->columns;
    int nBlocks = (size - 1) / BlockSize + 1;

    backpropSquaredClippedReluKernel<<<nBlocks, BlockSize>>>(unactivated->data, activated->data, gradients->data, size);

    cudaDeviceSynchronize();
}