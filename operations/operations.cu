#include "operations.h"

Matrix *Ones;
// Matrix is accessible from both cpu and gpu
// Some operations are written in CPU for simplicity, and other operations that
// need to be fast are written in GPU

// C= alpha * A + beta * B + theta
void linearFunction3Variable(Matrix *A, Matrix *B, Matrix *C, float alpha, float beta, float theta)
{
	assert(checkMemory(A) && checkMemory(B) && checkMemory(C));
	assert(checkDimension(A, B) && checkDimension(B, C));

	int size = A->rows * A->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	linearFunction3VariableKernel<<<nBlocks, BlockSize>>>(A->data, B->data, C->data, alpha, beta, theta, size);

	cudaDeviceSynchronize();
}
// B = alpha * A + beta
void linearFunction(Matrix *A, Matrix *B, float alpha, float beta)
{
	assert(checkMemory(A) && checkMemory(B));
	assert(checkDimension(A, B));

	int size = A->rows * A->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	linearFunctionKernel<<<nBlocks, BlockSize>>>(A->data, B->data, alpha, beta, size);

	cudaDeviceSynchronize();
}
// C = operation(A)*operation(B)
void MultipyMatrix(Matrix *A, Matrix *B, Matrix *C, int transpose_A, int transpose_B)
{
	assert(checkMemory(A) && checkMemory(B) && checkMemory(C));
	switch (2 * transpose_A + transpose_B)
	{
	case 0:
		assert(C->rows == A->rows && C->columns == B->columns && A->columns == B->rows);
		break;
	case 1:
		assert(C->rows == A->rows && C->columns == B->rows && A->columns == B->columns);
		break;
	case 2:
		assert(C->rows == A->columns && C->columns == B->columns && A->rows == B->rows);
		break;
	case 3:
		assert(C->rows == A->columns && C->columns == B->rows && A->rows == B->columns);
		break;
	default:
		break;
	}

	cublasHandle_t handle;
	cublasCreate(&handle);
	const float alpha = 1.0f;
	const float beta = 0.0f;
	// C = alpha*op( A )op( B ) + beta*C

	// switch order of multipication so that we can do multipication without transposing
	// This is needed because cublas is column-major ordered,and our matrices is row-major ordered.

	int m = C->columns;
	int n = C->rows;
	int k = transpose_A ? A->rows : A->columns;

	int lda = A->columns;
	int ldb = B->columns;
	int ldc = C->columns;
	cublasOperation_t trans_a = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t trans_b = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

	cublasSgemm(handle, trans_b, trans_a, m, n, k, &alpha, B->data, ldb, A->data, lda, &beta, C->data, ldc);
	cudaDeviceSynchronize();
	cublasDestroy(handle);
}

// sum rows of Matrix A and store the result in Matrix B
// if A = 8192 x 64 , B is 64 x 1
void sumRowsOfMatrix(Matrix *A, Matrix *B)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	const float alpha = 1.0f;
	const float beta = 0.0f;

	int m = A->columns;
	int n = A->rows;

	int lda = A->columns;
	// y = alpha* op(A)*x+ beta
	cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, A->data, lda, Ones->data, 1, &beta, B->data, 1);

	cudaDeviceSynchronize();
	cublasDestroy(handle);
}
// A = 0
void zeroMatrix(Matrix *A)
{
	linearFunction(A, A, 0.0f, 0.0f);
}
// A = fillValue
void fillMatrix(Matrix *A, float fillValue)
{
	linearFunction(A, A, 0.0f, fillValue);
}
// A = A + v
void addVectorToMatrix(Matrix *A, Matrix *v)
{
	assert(checkMemory(A) && checkMemory(v));

	int size = A->rows * A->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	addVectorToMatrixKernel<<<nBlocks, BlockSize>>>(A->data, v->data, A->columns, size);

	cudaDeviceSynchronize();
}
// limit values to [min, max]
void clipMatrix(Matrix *A, float min, float max)
{
	assert(checkMemory(A));

	int size = A->rows * A->columns;
	int nBlocks = (size - 1) / BlockSize + 1;

	clipMatrixKernel<<<nBlocks, BlockSize>>>(A->data, min, max, size);

	cudaDeviceSynchronize();
}
// Reshape matrix without changing its data
void reshapeMatrix(Matrix *A, int rows, int columns)
{
	assert(checkMemory(A));
	assert(A->rows * A->columns == rows * columns && rows > 0 && columns > 0);

	A->rows = rows;
	A->columns = columns;
}
// randomize Matrix with Gaussian distribution with a given mean and standard deviation
void randomizeMatrix(Matrix *A, float stddev, float mean)
{
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// https://github.com/AndyGrant/NNTrainer
#define uniform() ((double)(rand() + 1) / ((double)RAND_MAX + 2))
#define random() (sqrt(-2.0 * log(uniform())) * cos(2 * M_PI * uniform()))

	for (int j = 0; j < A->rows * A->columns; j++)
		A->data[j] = random() * stddev + mean;

#undef uniform
#undef random
}
// sum all elements of Matrix A and return the result
float sumMatrix(Matrix *A)
{
	float result = 0.0f;
	for (int i = 0; i < A->rows * A->columns; i++)
		result += A->data[i];
	return result;
}
// return the sparsity of Matrix
float sparsity(Matrix *A)
{
	int k = 0;
	for (int i = 0; i < A->rows * A->columns; i++)
	{
		if (A->data[i] != 0.0f)
			k++;
	}
	return 1.0 - k / (1.0 * A->rows * A->columns);
}
// return max element in Matrix A
float get_max_element(Matrix *A)
{
	assert(checkMemory(A));

	float max = A->data[0];
	for (int i = 0; i < A->rows * A->columns; i++)
	{
		max = MAX(max, A->data[i]);
	}
	return max;
}
// retur min element in Matrix A
float get_min_element(Matrix *A)
{
	assert(checkMemory(A));

	float min = A->data[0];
	for (int i = 0; i < A->rows * A->columns; i++)
	{
		min = MIN(min, A->data[i]);
	}
	return min;
}
void initOnes()
{
	Ones = createMatrix(BATCH_SIZE, 1);
	fillMatrix(Ones, 1.0f);
}