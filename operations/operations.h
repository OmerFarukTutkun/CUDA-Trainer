#ifndef _OPERATIONS_H_
#define _OPERATIONS_H_

#include "../defs.h"
#include "matrix.h"

void linearFunction3Variable(Matrix *A, Matrix *B, Matrix *C, float alpha, float beta, float theta);
void linearFunction(Matrix *A, Matrix *B, float alpha, float beta);
void fillMatrix(Matrix *A, float fillValue);
void zeroMatrix(Matrix *A);
void clipMatrix(Matrix *A, float min, float max);
void MultipyMatrix(Matrix *A, Matrix *B, Matrix *C, int transpose_A, int transpose_B);
void randomizeMatrix(Matrix *A, float stddev, float mean);
void reshapeMatrix(Matrix *A, int rows, int columns);
void sumRowsOfMatrix(Matrix *A, Matrix *B);
float get_max_element(Matrix *A);
float get_min_element(Matrix *A);
float sumMatrix(Matrix *A);
void addVectorToMatrix(Matrix *A, Matrix *v);
float sparsity(Matrix *A);
void initOnes();
// Kernels
__global__ void linearFunction3VariableKernel(float *A, float *B, float *C, float alpha, float beta, float theta, int N);
__global__ void linearFunctionKernel(float *A, float *B, float alpha, float beta, int N);
__global__ void clipMatrixKernel(float *A, float min, float max, int N);
__global__ void feature_transformer_slice_forward(int32_t *feature_indices, float *weight, float *bias, float *output);
__global__ void feature_transformer_slice_backward(int32_t *feature_indices, float *weight_grad, float *bias_grad, float *output_grad);
__global__ void addVectorToMatrixKernel(float *A, float *v, int m, int N);
__global__ void sumRowsOfMatrixKernel(float *A, float *B, int m, int N);

#endif