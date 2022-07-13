#ifndef _LOSS_H_
#define _LOSS_H_

#include "../defs.h"
#include "../operations/matrix.h"

typedef struct Loss
{
	void (*apply)(Matrix *, Matrix *, Matrix *);
	void (*backprop)(Matrix *, Matrix *, Matrix *);
} Loss;

void MSE(Matrix *prediction, Matrix *target, Matrix *loss);
void MAE(Matrix *prediction, Matrix *target, Matrix *loss);

void backpropMSE(Matrix *prediction, Matrix *target, Matrix *lossGradient);
void backpropMAE(Matrix *prediction, Matrix *target, Matrix *lossGradient);

__global__ void MSEKernel(float *prediction, float *target, float *loss, int N);
__global__ void MAEKernel(float *prediction, float *target, float *loss, int N);

__global__ void backpropMSEKernel(float *prediction, float *target, float *lossGradient, int N);
__global__ void backpropMAEKernel(float *prediction, float *target, float *lossGradient, int N);
#endif