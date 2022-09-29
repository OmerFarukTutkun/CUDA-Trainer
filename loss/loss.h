#ifndef _LOSS_H_
#define _LOSS_H_

#include "../defs.h"
#include "../operations/matrix.h"

typedef struct Loss
{
	void (*apply)(Matrix *, Matrix *, Matrix *);
	void (*backprop)(Matrix *, Matrix *, Matrix *);
} Loss;
extern Loss MSE;
extern Loss MAE;
void mse(Matrix *prediction, Matrix *target, Matrix *loss);
void mae(Matrix *prediction, Matrix *target, Matrix *loss);

void backpropMse(Matrix *prediction, Matrix *target, Matrix *lossGradient);
void backpropMae(Matrix *prediction, Matrix *target, Matrix *lossGradient);

__global__ void mseKernel(float *prediction, float *target, float *loss, int N);
__global__ void maeKernel(float *prediction, float *target, float *loss, int N);

__global__ void backpropMseKernel(float *prediction, float *target, float *lossGradient, int N);
__global__ void backpropMaeKernel(float *prediction, float *target, float *lossGradient, int N);

#endif