#include "matrix.h"

//create float Matrix with Cuda Malloc Managed so that the values are accessible both from CPU and GPU 
Matrix *createMatrix(int rows, int columns)
{
	assert(rows > 0 && columns > 0);
	float *data;

	Matrix *matrix;
	cudaMallocManaged(&matrix, sizeof(Matrix));
	cudaMallocManaged(&data, rows * columns * sizeof(float));

	assert(matrix != NULL && data != NULL);
	cudaMemset (data, 0 , sizeof(float) * rows*columns );
	matrix->data = data;
	matrix->rows = rows;
	matrix->columns = columns;
	return matrix;
}
void printMatrix(Matrix *matrix)
{
	assert(checkMemory(matrix));
	printf("Matrix with size %d x %d\n", matrix->rows, matrix->columns);

	if (matrix->rows > 40 || matrix->columns > 40)
	{
		printf("Matrix is too big to print\n");
	}
	for (int i = 0; i < MIN(40, matrix->rows); i++)
	{
		for (int j = 0; j < MIN(40, matrix->columns); j++)
		{
			printf("%10.5f  ", matrix->data[i * matrix->columns + j]);
		}
		printf("\n");
	}
}
void freeMatrix(Matrix *matrix)
{
	if (matrix == NULL || matrix->data == NULL)
		return;
	cudaFree(matrix->data);
	cudaFree(matrix);
}

int checkDimension(Matrix *a, Matrix *b)
{
	return (a->rows == b->rows && a->columns == b->columns);
}
int checkMemory(Matrix *a)
{
	return !(a == NULL || a->data == NULL);
}