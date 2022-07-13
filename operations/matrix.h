#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "../defs.h"

// rows-major ordered float matrix
typedef struct Matrix
{
	float *data;
	int rows, columns;
} Matrix;

Matrix *createMatrix(int rows, int columns);
void freeMatrix(Matrix *matrix);
int checkDimension(Matrix *a, Matrix *b);
int checkMemory(Matrix *a);
void printMatrix(Matrix *matrix);
#endif