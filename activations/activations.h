#ifndef _ACTIVATIONS_H_
#define _ACTIVATIONS_H_
#include "../defs.h"
#include "../operations/matrix.h"

#define ClippedReluMax 1.0f

typedef struct Activation
{
	void (*apply)(Matrix *, Matrix *);
	void (*backprop)(Matrix *, Matrix *, Matrix *);
} Activation;

// activation functions
void relu(Matrix *unactivated, Matrix *activated);
void sigmoid(Matrix *unactivated, Matrix *activated);
void clippedRelu(Matrix *unactivated, Matrix *activated);

// multipy gradient vector by gradients of activation functions in backpropagation
void backpropRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients);
void backpropSigmoid(Matrix *unactivated, Matrix *activated, Matrix *gradients);
void backpropClippedRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients);

// Kernels
__global__ void reluKernel(float *unactivated, float *activated, int N);
__global__ void sigmoidKernel(float *unactivated, float *activated, int N);
__global__ void clippedReluKernel(float *unactivated, float *activated, int N);

__global__ void backpropReluKernel(float *unactivated, float *activated, float *gradients, int N);
__global__ void backpropSigmoidKernel(float *unactivated, float *activated, float *gradients, int N);
__global__ void backpropClippedReluKernel(float *unactivated, float *activated, float *gradients, int N);

#endif