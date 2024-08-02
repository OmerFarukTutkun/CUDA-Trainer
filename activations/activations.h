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

extern Activation Relu ;
extern Activation ClippedRelu ;
extern Activation Sigmoid;
extern Activation SquaredClippedRelu;

// activation functions
void squaredClippedRelu(Matrix *unactivated, Matrix *activated);
void relu(Matrix *unactivated, Matrix *activated);
void sigmoid(Matrix *unactivated, Matrix *activated);
void clippedRelu(Matrix *unactivated, Matrix *activated);

// multipy gradient vector by gradients of activation functions in backpropagation
void backpropSquaredClippedRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients);
void backpropRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients);
void backpropSigmoid(Matrix *unactivated, Matrix *activated, Matrix *gradients);
void backpropClippedRelu(Matrix *unactivated, Matrix *activated, Matrix *gradients);

// Kernels
__global__ void squaredClippedReluKernel(float *unactivated, float *activated, int N);
__global__ void reluKernel(float *unactivated, float *activated, int N);
__global__ void sigmoidKernel(float *unactivated, float *activated, int N);
__global__ void clippedReluKernel(float *unactivated, float *activated, int N);

__global__ void backpropSquaredClippedReluKernel(float *unactivated, float *activated, float *gradients, int N);
__global__ void backpropReluKernel(float *unactivated, float *activated, float *gradients, int N);
__global__ void backpropSigmoidKernel(float *unactivated, float *activated, float *gradients, int N);
__global__ void backpropClippedReluKernel(float *unactivated, float *activated, float *gradients, int N);


#endif